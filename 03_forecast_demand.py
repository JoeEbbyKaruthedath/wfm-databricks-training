# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Forecast demand with the champion model
# MAGIC
# MAGIC This notebook covers **model as code**: the code ships through Git and jobs, and it retrieves whichever model
# MAGIC version is currently `@champion` from Unity Catalog.
# MAGIC - Load `models:/<catalog>.<schema>.<model>@champion`
# MAGIC - Score every venue and open hour for the coming days across the cluster with `mlflow.pyfunc.spark_udf`
# MAGIC - Write the forecast to a Unity Catalog table that the optimizer (notebook 04) reads
# MAGIC
# MAGIC In production this is the first task of the **nightly job**.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.text("model_name", "venue_demand_forecaster", "Model name")
dbutils.widgets.text("forecast_days", "7", "Days to forecast")
dbutils.widgets.text("start_date", "", "First forecast date (blank = tomorrow)")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
forecast_days = int(dbutils.widgets.get("forecast_days"))
start_date = dbutils.widgets.get("start_date").strip()

full_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

from datetime import date, timedelta

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from wfm_utils import FEATURES, build_feature_frame

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
# MAGIC ## Score the coming days for every venue

# COMMAND ----------

start = date.fromisoformat(start_date) if start_date else date.today() + timedelta(days=1)
dates = [start + timedelta(days=i) for i in range(forecast_days)]

venues_pdf = spark.table(f"{catalog}.{schema}.venues").toPandas()
grid = spark.createDataFrame(build_feature_frame(venues_pdf, dates))

forecast = (
    grid.withColumn("forecast_guests", F.round(F.greatest(F.lit(0.0), predict(F.struct(*FEATURES))), 1))
    .withColumn("model_version", F.lit(int(champion.version)))
    .withColumn("forecast_created_at", F.current_timestamp())
    .select("venue_id", "park", "venue_type", "date", "hour", "forecast_guests", "model_version", "forecast_created_at")
)

forecast_table = f"{catalog}.{schema}.venue_demand_forecast"
forecast.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(forecast_table)
print(f"Wrote {spark.table(forecast_table).count():,} rows to {forecast_table} for {dates[0]} to {dates[-1]}")

# COMMAND ----------

display(spark.sql(f"""
    SELECT date, venue_type, ROUND(SUM(forecast_guests)) AS forecast_guests
    FROM {forecast_table}
    GROUP BY date, venue_type
    ORDER BY date, venue_type
"""))

# COMMAND ----------

dbutils.notebook.exit(f"model_version={champion.version}, start={dates[0]}, days={forecast_days}")
