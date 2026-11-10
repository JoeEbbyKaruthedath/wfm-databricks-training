# Databricks notebook source
# MAGIC %pip install "ortools>=9.10,<10" --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 · Build staff schedules with CP-SAT, venues in parallel
# MAGIC
# MAGIC This notebook covers:
# MAGIC - **Notebook-scoped libraries:** the `%pip install` cell above installs OR-Tools for this notebook only
# MAGIC - **Spark for parallel work:** `groupBy("venue_id").applyInPandas(...)` hands each venue's data to a Python function on the cluster, so venues are solved in parallel
# MAGIC - **Outputs:** the shift schedules and a staffing summary, as tables in Unity Catalog
# MAGIC
# MAGIC In production this is the second task of the **nightly job**. For daytime reruns, run the job with
# MAGIC different parameters (for example, a higher `service_level` or a shorter `max_shift_hours`).

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.text("venues", "all", "Venues (all, or comma-separated IDs)")
dbutils.widgets.text("service_level", "1.0", "Service level (1.1 = 10% more staff)")
dbutils.widgets.text("min_shift_hours", "4", "Shortest shift (hours)")
dbutils.widgets.text("max_shift_hours", "8", "Longest shift (hours)")
dbutils.widgets.text("solver_time_limit_s", "20", "Solver time limit per venue (s)")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")
venues = dbutils.widgets.get("venues").strip()
service_level = float(dbutils.widgets.get("service_level"))
min_shift = int(dbutils.widgets.get("min_shift_hours"))
max_shift = int(dbutils.widgets.get("max_shift_hours"))
time_limit = float(dbutils.widgets.get("solver_time_limit_s"))

SHIFT_LENGTHS = [h for h in (4, 6, 8, 10) if min_shift <= h <= max_shift]
SOLVER_THREADS = 1  # CP-SAT uses every core on the machine by default; keep it to the cores each Spark task gets

# COMMAND ----------

# MAGIC %md
# MAGIC ## Staff needed per hour
# MAGIC Required staff for each venue, role and hour = forecast guests × service level ÷ guests one person can serve,
# MAGIC with a minimum per role.

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

forecast = spark.table(f"{catalog}.{schema}.venue_demand_forecast")
venue_info = spark.table(f"{catalog}.{schema}.venues").select("venue_id", "venue_name", "open_hour", "close_hour")
roles = spark.table(f"{catalog}.{schema}.venue_roles")

required = (
    forecast.join(venue_info, "venue_id")
    .join(roles, "venue_type")
    .withColumn(
        "required_staff",
        F.greatest(F.col("min_staff"), F.ceil(F.col("forecast_guests") * service_level / F.col("guests_per_staff_hour"))).cast("int"),
    )
    .select("venue_id", "venue_name", "park", "venue_type", "open_hour", "close_hour", "date", "hour", "role", "required_staff")
)
if venues.lower() != "all":
    required = required.where(F.col("venue_id").isin([v.strip() for v in venues.split(",")]))

display(required.where(F.col("venue_id") == required.first()["venue_id"]).orderBy("date", "role", "hour").limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The optimizer for one venue
# MAGIC For every day and role: choose how many people start a shift at each hour, and for how long, so that every
# MAGIC open hour has at least the required staff, at the lowest total staff-hours (plus a small cost per shift).
# MAGIC Each day and role is its own small CP-SAT model; one Spark task solves all of them for its venue.

# COMMAND ----------

schedule_schema = (
    "venue_id string, venue_name string, date date, role string, shift_start int, shift_end int, "
    "shift_hours int, staff_count int, solver_status string, solve_seconds double"
)


def solve_venue(pdf: pd.DataFrame) -> pd.DataFrame:
    import time

    from ortools.sat.python import cp_model

    venue_id, venue_name = pdf["venue_id"].iloc[0], pdf["venue_name"].iloc[0]
    open_hour, close_hour = int(pdf["open_hour"].iloc[0]), int(pdf["close_hour"].iloc[0])
    rows, statuses, t0 = [], set(), time.time()

    for (day, role), g in pdf.groupby(["date", "role"]):
        need = dict(zip(g["hour"].astype(int), g["required_staff"].astype(int)))
        model = cp_model.CpModel()
        options = [
            (start, length, model.new_int_var(0, max(need.values()), f"start{start}_len{length}"))
            for start in range(open_hour, close_hour)
            for length in SHIFT_LENGTHS
            if start + length <= close_hour
        ]
        for hour in range(open_hour, close_hour):
            model.add(sum(x for s, L, x in options if s <= hour < s + L) >= need.get(hour, 0))
        # Cost in half-hours: staff-hours worked plus half an hour per shift for handover.
        model.minimize(sum(x * (2 * L + 1) for s, L, x in options))

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = SOLVER_THREADS
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.solve(model)
        statuses.add(solver.status_name(status))
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            rows += [(day, role, s, s + L, L, int(solver.value(x))) for s, L, x in options if solver.value(x) > 0]

    overall = "OPTIMAL" if statuses == {"OPTIMAL"} else ",".join(sorted(statuses))
    elapsed = round(time.time() - t0, 3)
    out = pd.DataFrame(rows, columns=["date", "role", "shift_start", "shift_end", "shift_hours", "staff_count"])
    out.insert(0, "venue_name", venue_name)
    out.insert(0, "venue_id", venue_id)
    out["solver_status"], out["solve_seconds"] = overall, elapsed
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run every venue in parallel
# MAGIC `applyInPandas` groups the rows by venue and spreads the venues across Spark tasks, which run in parallel on the
# MAGIC cluster's cores. Each task solves the venues in its partition. Two settings keep the venues spread across many tasks:
# MAGIC enough shuffle partitions, and no merging of small partitions (each venue's input is tiny, but the solve is not).

# COMMAND ----------

n_venues = required.select("venue_id").distinct().count()
spark.conf.set("spark.sql.shuffle.partitions", str(max(n_venues, 8)))
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

schedules = required.groupBy("venue_id").applyInPandas(solve_venue, schema=schedule_schema)

schedule_table = f"{catalog}.{schema}.venue_staff_schedule"
schedules.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(schedule_table)
display(spark.table(schedule_table).orderBy("venue_id", "date", "role", "shift_start"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check coverage
# MAGIC Scheduled staff on duty vs. required staff for every venue, role and hour (saved as `venue_staffing_coverage`),
# MAGIC and a summary by venue (saved as `venue_staffing_summary`).

# COMMAND ----------

sched_pdf = spark.table(schedule_table).toPandas()
req_pdf = required.toPandas()

on_duty = (
    sched_pdf.assign(hour=lambda d: [list(range(s, e)) for s, e in zip(d["shift_start"], d["shift_end"])])
    .explode("hour")
    .astype({"hour": int})
    .groupby(["venue_id", "date", "role", "hour"], as_index=False)["staff_count"].sum()
    .rename(columns={"staff_count": "scheduled_staff"})
)
coverage = req_pdf.merge(on_duty, on=["venue_id", "date", "role", "hour"], how="left").fillna({"scheduled_staff": 0})
coverage["scheduled_staff"] = coverage["scheduled_staff"].astype(int)
coverage["understaffed"] = (coverage["scheduled_staff"] < coverage["required_staff"]).astype(int)

coverage_table = f"{catalog}.{schema}.venue_staffing_coverage"
(spark.createDataFrame(coverage.drop(columns=["open_hour", "close_hour"]))
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(coverage_table))

solver_stats = sched_pdf.groupby("venue_id", as_index=False).agg(
    shifts=("staff_count", "sum"), solver_status=("solver_status", "first"), solve_seconds=("solve_seconds", "first")
)
summary = (
    coverage.groupby(["venue_id", "venue_name"], as_index=False)
    .agg(required_staff_hours=("required_staff", "sum"), scheduled_staff_hours=("scheduled_staff", "sum"),
         understaffed_hours=("understaffed", "sum"))
    .merge(solver_stats, on="venue_id")
)
summary["overstaffing_pct"] = (summary["scheduled_staff_hours"] / summary["required_staff_hours"] - 1).round(3)

summary_table = f"{catalog}.{schema}.venue_staffing_summary"
spark.createDataFrame(summary).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(summary_table)
display(summary)

# COMMAND ----------

dbutils.notebook.exit(f"venues={len(summary)}, shifts={int(summary['shifts'].sum())}")
