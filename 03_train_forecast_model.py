# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train the demand forecaster with MLflow
# MAGIC
# MAGIC This notebook covers:
# MAGIC - **Experiments:** every run records its parameters, metrics, plots and the exact training data
# MAGIC - **Model registry in Unity Catalog:** each training run registers a new, versioned model
# MAGIC - **Champion/challenger:** the new version becomes `@champion` only if it beats the current champion on the holdout weeks
# MAGIC
# MAGIC This is the second task of the **training** pipeline, after `02_feature_engineering` builds `venue_training_features`.
# MAGIC The inference pipeline always loads `@champion`.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.text("model_name", "venue_demand_forecaster", "Model name")
dbutils.widgets.text("n_estimators", "300", "XGBoost: trees")
dbutils.widgets.text("max_depth", "6", "XGBoost: max depth")
dbutils.widgets.text("learning_rate", "0.1", "XGBoost: learning rate")
dbutils.widgets.text("holdout_weeks", "8", "Holdout weeks")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
n_estimators = int(dbutils.widgets.get("n_estimators"))
max_depth = int(dbutils.widgets.get("max_depth"))
learning_rate = float(dbutils.widgets.get("learning_rate"))
holdout_weeks = int(dbutils.widgets.get("holdout_weeks"))

full_model_name = f"{catalog}.{schema}.{model_name}"
features_table = f"{catalog}.{schema}.venue_training_features"

# COMMAND ----------

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor

from wfm_utils import FEATURES

mlflow.set_registry_uri("databricks-uc")  # register models in Unity Catalog
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/wfm_training_demand_forecast")  # outside the Git folder: experiments can't live in one
client = MlflowClient()
print("MLflow", mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training data
# MAGIC The features come from `venue_training_features`. The last few weeks are held out so every model version is
# MAGIC scored on the same unseen period.

# COMMAND ----------

training_data = spark.table(features_table)
features_version = spark.sql(f"DESCRIBE HISTORY {features_table} LIMIT 1").first()["version"]

pdf = training_data.toPandas()
pdf["date"] = pd.to_datetime(pdf["date"])
cutoff = pdf["date"].max() - pd.Timedelta(weeks=holdout_weeks)
train, test = pdf[pdf["date"] <= cutoff], pdf[pdf["date"] > cutoff]
X_train, y_train = train[FEATURES], train["guests"]
X_test, y_test = test[FEATURES], test["guests"]
print(f"train: {len(train):,} rows up to {cutoff.date()} | holdout: {len(test):,} rows")


def score(y_true, y_pred):
    """MAPE on busy hours (10+ guests) and weighted absolute percentage error overall."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.clip(np.asarray(y_pred, dtype=float), 0, None)
    busy = y_true >= 10
    mape = float(np.mean(np.abs(y_pred[busy] - y_true[busy]) / y_true[busy]))
    wape = float(np.abs(y_pred - y_true).sum() / y_true.sum())
    return mape, wape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and track the run
# MAGIC - `autolog` records parameters, training metrics and feature importance.
# MAGIC - `log_input` records the exact table **and Delta version** the model was trained on.
# MAGIC - The model is logged with a signature, then registered in Unity Catalog. No pickle files to manage.

# COMMAND ----------

mlflow.xgboost.autolog(log_models=False)  # we log and register the model explicitly below

with mlflow.start_run(run_name=f"xgb_depth{max_depth}_trees{n_estimators}") as run:
    mlflow.log_input(
        mlflow.data.from_spark(training_data, table_name=features_table, version=str(features_version)),
        context="training",
    )
    mlflow.log_params({"holdout_weeks": holdout_weeks, "training_cutoff": str(cutoff.date())})

    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    mape, wape = score(y_test, model.predict(X_test))
    mlflow.log_metrics({"holdout_mape": mape, "holdout_wape": wape})

    # One venue's last holdout week: forecast vs. actual
    venue = test["venue_id"].iloc[0]
    week = test[(test["venue_id"] == venue) & (test["date"] > test["date"].max() - pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(week["guests"].to_numpy(), label="actual")
    ax.plot(model.predict(week[FEATURES]).clip(0), label="forecast")
    ax.set_title(f"{venue}: last holdout week (hourly)")
    ax.set_xlabel("open hour")
    ax.set_ylabel("guests")
    ax.legend()
    mlflow.log_figure(fig, "holdout_forecast_vs_actual.png")

    signature = infer_signature(X_train.head(100), model.predict(X_train.head(100)))
    model_info = mlflow.xgboost.log_model(model, name="model", signature=signature, input_example=X_train.head(5))

new_version = mlflow.register_model(model_info.model_uri, full_model_name).version
print(f"Run {run.info.run_id}: holdout MAPE {mape:.1%}, WAPE {wape:.1%} → registered {full_model_name} v{new_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion or challenger?
# MAGIC Score the current `@champion` on the same holdout weeks. Promote the new version only if it's better.
# MAGIC Promotion just moves the alias, so the inference pipeline picks up the new model with no code change.

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(full_model_name, "champion")
    champion_mape, _ = score(y_test, mlflow.pyfunc.load_model(f"models:/{full_model_name}@champion").predict(X_test))
except mlflow.exceptions.MlflowException:
    champion, champion_mape = None, None

if champion is None or mape < champion_mape:
    if champion is not None:
        client.set_registered_model_alias(full_model_name, "previous_champion", champion.version)
    client.set_registered_model_alias(full_model_name, "champion", new_version)
    decision = "promoted to @champion"
else:
    client.set_registered_model_alias(full_model_name, "challenger", new_version)
    decision = "kept the current champion; new version is @challenger"

summary = {
    "model": full_model_name,
    "new_version": int(new_version),
    "new_holdout_mape": round(mape, 4),
    "champion_version_before": int(champion.version) if champion else None,
    "champion_holdout_mape": round(champion_mape, 4) if champion_mape is not None else None,
    "decision": decision,
}
display(pd.DataFrame([summary]))

# COMMAND ----------

dbutils.notebook.exit(json.dumps(summary))
