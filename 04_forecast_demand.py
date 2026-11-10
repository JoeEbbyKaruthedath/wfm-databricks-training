# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Forecast demand with the champion model (inference)
# MAGIC
# MAGIC This notebook covers **model as code**: the code ships through Git and the inference pipeline, and it retrieves
# MAGIC whichever model version is currently `@champion` from Unity Catalog.
# MAGIC - Load `models:/<catalog>.<schema>.<model>@champion`
# MAGIC - Score the features in `venue_scoring_features` (built by `02_feature_engineering`) across the cluster with
# MAGIC   `mlflow.pyfunc.spark_udf`
# MAGIC - Write the forecast to `venue_demand_forecast`, which the optimizer (`05_optimize_staff_schedules`) reads
# MAGIC
# MAGIC This is the second task of the **inference** pipeline.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.text("model_name", "venue_demand_forecaster", "Model name")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

full_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from wfm_utils import FEATURES

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve the champion model

# COMMAND ----------

champion = client.get_model_version_by_alias(full_model_name, "champion")
model_uri = f"models:/{full_model_name}/{champion.version}"  # pin the exact version the alias points to right now
print(f"Using {full_model_name} v{champion.version} (@champion)")

predict = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double", env_manager="local")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score every venue and open hour in the forecast window

# COMMAND ----------

features = spark.table(f"{catalog}.{schema}.venue_scoring_features")

forecast = (
    features.withColumn("forecast_guests", F.round(F.greatest(F.lit(0.0), predict(F.struct(*FEATURES))), 1))
    .withColumn("model_version", F.lit(int(champion.version)))
    .withColumn("forecast_created_at", F.current_timestamp())
    .select("venue_id", "park", "venue_type", "date", "hour", "forecast_guests", "model_version", "forecast_created_at")
)

forecast_table = f"{catalog}.{schema}.venue_demand_forecast"
forecast.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(forecast_table)
window = spark.table(forecast_table).agg(F.min("date").alias("first"), F.max("date").alias("last"), F.count("*").alias("n")).first()
print(f"Wrote {window['n']:,} rows to {forecast_table} for {window['first']} to {window['last']}")

# COMMAND ----------

display(spark.sql(f"""
    SELECT date, venue_type, ROUND(SUM(forecast_guests)) AS forecast_guests
    FROM {forecast_table}
    GROUP BY date, venue_type
    ORDER BY date, venue_type
"""))

# COMMAND ----------

dbutils.notebook.exit(f"model_version={champion.version}, start={window['first']}, end={window['last']}")
