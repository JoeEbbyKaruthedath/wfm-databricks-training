# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Create the jobs (jobs as code)
# MAGIC
# MAGIC Creates or updates the three training jobs, pointing at the notebooks in **this folder**:
# MAGIC - **WFM training: Create seed data** runs `01`
# MAGIC - **WFM training: Train demand forecaster** runs `02` weekly (schedule paused)
# MAGIC - **WFM training: Nightly forecast and staff schedules** runs `03`, then `04` (schedule paused)
# MAGIC
# MAGIC Run it once in a new workspace. Running it again updates the jobs in place (it matches them by name).
# MAGIC Everything here can also be done by hand in **Jobs & Pipelines → Create → Job**. This notebook just writes the
# MAGIC same settings down as code, using the Databricks SDK that comes with every runtime.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "wfm_training", "Schema")
dbutils.widgets.text("node_type_id", "", "Node type (blank = pick 4 cores, 16 GB)")
dbutils.widgets.text("spark_version", "", "Runtime (blank = latest ML LTS)")
dbutils.widgets.text("notify_email", "", "Failure emails to (blank = you)")

catalog = dbutils.widgets.get("catalog").strip()
if not catalog:
    raise ValueError("Set the catalog widget to a Unity Catalog catalog where you can create a schema.")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

import os
import re

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me().user_name
notify = dbutils.widgets.get("notify_email").strip() or me
folder = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())


def latest_ml_lts():
    best = None
    for v in w.clusters.spark_versions().versions:
        name, key = v.name, v.key
        if "LTS" in name and " ML" in name and not re.search(r"GPU|Photon|Beta|aarch64", name + key):
            ver = tuple(int(x) for x in re.findall(r"^(\d+)\.(\d+|x)", key)[0] if x.isdigit())
            if best is None or ver > best[0]:
                best = (ver, key)
    return best[1]


spark_version = dbutils.widgets.get("spark_version").strip() or latest_ml_lts()
def general_purpose_node():
    """Smallest general-purpose (non-GPU) node with 4 cores and 16 GB, preferring one with local disk."""
    for local_disk in (True, None):
        try:
            return w.clusters.select_node_type(category="General Purpose", min_cores=4, min_memory_gb=16, local_disk=local_disk)
        except ValueError:
            continue
    raise ValueError("No general-purpose node with 4 cores and 16 GB found; set the node_type_id widget.")


node_type = dbutils.widgets.get("node_type_id").strip() or general_purpose_node()
print(f"Notebooks: {folder}\nRuntime:   {spark_version}\nNode type: {node_type}\nAlerts to: {notify}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Job definitions
# MAGIC - **Job compute:** each job creates its own cluster per run, which shuts down when the run finishes.
# MAGIC - **Single node** for seed data, training and forecasting; a small **autoscaling** cluster for the optimizer.
# MAGIC - **Job parameters** fill the notebook widgets of every task.

# COMMAND ----------

single_node = {
    "spark_version": spark_version, "node_type_id": node_type, "num_workers": 0, "data_security_mode": "SINGLE_USER",
    "spark_conf": {"spark.databricks.cluster.profile": "singleNode", "spark.master": "local[*, 4]"},
    "custom_tags": {"ResourceClass": "SingleNode", "demo": "wfm-training"},
}
autoscale = {
    "spark_version": spark_version, "node_type_id": node_type, "autoscale": {"min_workers": 1, "max_workers": 3},
    "data_security_mode": "SINGLE_USER", "custom_tags": {"demo": "wfm-training"},
}
common = {"email_notifications": {"on_failure": [notify]}, "max_concurrent_runs": 1, "tags": {"demo": "wfm-training"}}


def param(name, default):
    return {"name": name, "default": default}


def notebook_task(key, notebook, cluster, depends_on=None, retries=1):
    task = {
        "task_key": key, "job_cluster_key": cluster, "timeout_seconds": 3600,
        "notebook_task": {"notebook_path": f"{folder}/{notebook}", "source": "WORKSPACE"},
    }
    if retries:
        task |= {"max_retries": retries, "min_retry_interval_millis": 60000}
    if depends_on:
        task["depends_on"] = [{"task_key": depends_on}]
    return task


base = [param("catalog", catalog), param("schema", schema)]
jobs = [
    {**common, "name": "WFM training: Create seed data",
     "description": "Creates the schema, the exports volume and the synthetic seed data. Safe to rerun; it overwrites the seed tables.",
     "parameters": base + [param("history_days", "730")],
     "job_clusters": [{"job_cluster_key": "single_node", "new_cluster": single_node}],
     "tasks": [notebook_task("create_seed_data", "01_setup_and_data", "single_node", retries=0)]},
    {**common, "name": "WFM training: Train demand forecaster",
     "description": "Trains the demand model, registers a new version in Unity Catalog, and promotes it to @champion only if it beats the current champion.",
     "parameters": base + [param("model_name", "venue_demand_forecaster"), param("n_estimators", "300"),
                           param("max_depth", "6"), param("learning_rate", "0.1")],
     "job_clusters": [{"job_cluster_key": "single_node", "new_cluster": single_node}],
     "tasks": [notebook_task("train_and_promote", "02_train_forecast_model", "single_node")],
     "schedule": {"quartz_cron_expression": "0 0 2 ? * MON", "timezone_id": "America/New_York", "pause_status": "PAUSED"}},
    {**common, "name": "WFM training: Nightly forecast and staff schedules",
     "description": "Forecasts demand with the @champion model, then builds staff schedules for every venue.",
     "parameters": base + [param("model_name", "venue_demand_forecaster"), param("forecast_days", "7"), param("start_date", ""),
                           param("venues", "all"), param("service_level", "1.0"), param("min_shift_hours", "4"),
                           param("max_shift_hours", "8")],
     "job_clusters": [{"job_cluster_key": "forecast_single_node", "new_cluster": single_node},
                      {"job_cluster_key": "optimize_autoscale", "new_cluster": autoscale}],
     "tasks": [notebook_task("forecast_demand", "03_forecast_demand", "forecast_single_node"),
               notebook_task("optimize_schedules", "04_optimize_staff_schedules", "optimize_autoscale", depends_on="forecast_demand")],
     "schedule": {"quartz_cron_expression": "0 0 1 * * ?", "timezone_id": "America/New_York", "pause_status": "PAUSED"}},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update

# COMMAND ----------

host = w.config.host.rstrip("/")
existing = {j.settings.name: j.job_id for j in w.jobs.list() if j.settings and j.settings.name in {s["name"] for s in jobs}}

for settings in jobs:
    job_id = existing.get(settings["name"])
    if job_id:
        w.api_client.do("POST", "/api/2.2/jobs/reset", body={"job_id": job_id, "new_settings": settings})
        action = "updated"
    else:
        job_id = w.api_client.do("POST", "/api/2.2/jobs/create", body=settings)["job_id"]
        action = "created"
    print(f"{action}: {settings['name']} → {host}/jobs/{job_id}")
