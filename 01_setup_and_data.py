# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Workspace basics, Unity Catalog and the demo data
# MAGIC
# MAGIC This notebook covers:
# MAGIC - **Notebook basics:** widgets (parameters), `display()`, and importing a `.py` file stored next to this notebook
# MAGIC - **Unity Catalog:** `catalog.schema.table`, tables vs. volumes
# MAGIC - **The demo data:** 20 synthetic venues across two parks, with two years of hourly guest demand
# MAGIC
# MAGIC **Run order:** `01` once to set up → `02` train the forecaster → `03` forecast demand → `04` build staff schedules.
# MAGIC
# MAGIC All data in this demo is synthetic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC Widgets are notebook parameters. When this notebook runs as a job task, job parameters fill the same widgets.
# MAGIC Changing `catalog` and `schema` is all it takes to run the same code in dev, pre-prod or prod.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.text("history_days", "730", "Days of history")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")
history_days = int(dbutils.widgets.get("history_days"))
print(f"Writing to {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog: schema and volume
# MAGIC - **Tables** hold structured data (demand history, forecasts, schedules).
# MAGIC - **Volumes** hold files, such as CSV extracts, images or documents, governed by the same permissions as tables.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}` COMMENT 'WFM training demo (synthetic data)'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.exports COMMENT 'Files shared with the team'")
display(spark.sql(f"SHOW TABLES IN `{catalog}`.`{schema}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading from Snowflake (pattern only, not run here)
# MAGIC In production the inputs come from Snowflake through the connector and a service principal. Push filters and
# MAGIC column lists into the query instead of `SELECT *`, because the Snowflake-to-Databricks transfer rate is fixed.
# MAGIC
# MAGIC ```python
# MAGIC sf_options = {
# MAGIC     "sfUrl": "<account>.snowflakecomputing.com",
# MAGIC     "sfUser": dbutils.secrets.get("wfm", "sf_client_id"),
# MAGIC     "pem_private_key": dbutils.secrets.get("wfm", "sf_private_key"),
# MAGIC     "sfWarehouse": "<warehouse>", "sfDatabase": "<db>", "sfSchema": "<schema>",
# MAGIC }
# MAGIC demand = (spark.read.format("snowflake").options(**sf_options)
# MAGIC     .option("query", """
# MAGIC         SELECT venue_id, ts, guests
# MAGIC         FROM venue_transactions
# MAGIC         WHERE ts >= DATEADD(year, -1, CURRENT_DATE)
# MAGIC     """)
# MAGIC     .load())
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the demo data
# MAGIC The generator lives in `wfm_utils.py`, a plain Python file next to this notebook. Notebooks can import it like any module.

# COMMAND ----------

from wfm_utils import build_venues, build_roles, generate_demand_history

venues_pdf = build_venues()
roles_pdf = build_roles()
history_pdf = generate_demand_history(venues_pdf, days=history_days)

display(venues_pdf)

# COMMAND ----------

def save(pdf, table, comment):
    name = f"`{catalog}`.`{schema}`.{table}"
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
    spark.sql(f"COMMENT ON TABLE {name} IS '{comment}'")
    print(f"{name}: {len(pdf):,} rows")

save(venues_pdf, "venues", "Synthetic venues: park, type, base demand and opening hours")
save(roles_pdf, "venue_roles", "Staffing roles per venue type: guests one staff member can serve per hour, minimum staff")
save(history_pdf, "venue_demand_history", "Synthetic hourly guest demand per venue")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A quick look at the history
# MAGIC Use the chart button on the result to plot average guests by hour for each venue type.

# COMMAND ----------

display(spark.sql(f"""
    SELECT venue_type, hour, ROUND(AVG(guests)) AS avg_guests
    FROM `{catalog}`.`{schema}`.venue_demand_history
    GROUP BY venue_type, hour
    ORDER BY venue_type, hour
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: explore in Catalog Explorer
# MAGIC - **Overview and sample data** for `venue_demand_history`
# MAGIC - **Permissions:** one set of grants covers tables, volumes and models, for example
# MAGIC   `GRANT SELECT ON SCHEMA <catalog>.wfm_training TO <group>`
# MAGIC - **Lineage:** after notebooks 02–04 run, it shows which notebook or job wrote each table
