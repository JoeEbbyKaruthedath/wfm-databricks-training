# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Feature engineering
# MAGIC
# MAGIC Builds the model features from the raw data. Training and scoring use this same code, so the model always sees
# MAGIC features computed the same way.
# MAGIC - `purpose = training`: one row per venue, day and open hour of the history, with the actual guests as the label.
# MAGIC   Writes `venue_training_features`.
# MAGIC - `purpose = scoring`: one row per venue, day and open hour for the coming days. Writes `venue_scoring_features`.
# MAGIC
# MAGIC **Features:** venue size (`base_demand`), venue type, park, hour, day of week, month, peak season, holiday, and
# MAGIC evening event.
# MAGIC
# MAGIC This is the first task of both the **training** and the **inference** pipelines.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.dropdown("purpose", "training", ["training", "scoring"], "Purpose")
dbutils.widgets.text("forecast_days", "7", "Scoring: days to forecast")
dbutils.widgets.text("start_date", "", "Scoring: first date (blank = tomorrow)")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")
purpose = dbutils.widgets.get("purpose")
forecast_days = int(dbutils.widgets.get("forecast_days"))
start_date = dbutils.widgets.get("start_date").strip()

# COMMAND ----------

from datetime import date, timedelta

from pyspark.sql import functions as F

from wfm_utils import FEATURES, calendar_features

venues = spark.table(f"{catalog}.{schema}.venues").select(
    "venue_id", "park", "venue_type", "park_code", "venue_type_code", "base_demand", "open_hour", "close_hour"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rows to build features for
# MAGIC - **Training:** every venue-hour in the raw demand history.
# MAGIC - **Scoring:** every venue, day and open hour in the forecast window, generated with Spark.

# COMMAND ----------

if purpose == "training":
    rows = spark.table(f"{catalog}.{schema}.venue_demand_history")  # venue_id, date, hour, guests
    dates = [r["date"] for r in rows.select("date").distinct().collect()]
    target_table = "venue_training_features"
elif purpose == "scoring":
    start = date.fromisoformat(start_date) if start_date else date.today() + timedelta(days=1)
    dates = [start + timedelta(days=i) for i in range(forecast_days)]
    rows = (
        venues.select("venue_id", "open_hour", "close_hour")
        .crossJoin(spark.createDataFrame([(d,) for d in dates], "date date"))
        .withColumn("hour", F.explode(F.sequence(F.col("open_hour"), F.col("close_hour") - 1)))
        .drop("open_hour", "close_hour")
    )
    target_table = "venue_scoring_features"
else:
    raise ValueError(f"purpose must be 'training' or 'scoring', not {purpose!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add venue and calendar features
# MAGIC Calendar features come from `calendar_features()` in `wfm_utils.py`: day of week, month, peak season, holidays and
# MAGIC evening-event nights.

# COMMAND ----------

calendar = spark.createDataFrame(calendar_features(dates))
label = ["guests"] if purpose == "training" else []

features = (
    rows.join(venues.drop("open_hour", "close_hour"), "venue_id")
    .join(calendar, "date")
    .withColumn("hour", F.col("hour").cast("long"))
    .select("venue_id", "park", "venue_type", "date", *FEATURES, *label)
)

table = f"{catalog}.{schema}.{target_table}"
features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
n_rows = spark.table(table).count()
print(f"{table}: {n_rows:,} rows for {min(dates)} to {max(dates)}")
display(spark.table(table).limit(20))

# COMMAND ----------

dbutils.notebook.exit(f"{target_table}: {n_rows} rows, {min(dates)} to {max(dates)}")
