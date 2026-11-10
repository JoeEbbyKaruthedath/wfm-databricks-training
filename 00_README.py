# Databricks notebook source
# MAGIC %md
# MAGIC # WFM training: start here
# MAGIC
# MAGIC A small, end-to-end version of the workforce management (WFM) pipeline on Databricks, built on **synthetic data**:
# MAGIC forecast hourly guest demand per venue, then build optimized staff schedules per venue.
# MAGIC Follow along during the session, or work through the steps below on your own afterwards.
# MAGIC
# MAGIC | Notebook | What it does | Runs as |
# MAGIC |---|---|---|
# MAGIC | `01_setup_and_data` | Creates the schema, the `exports` volume and the seed data: 20 venues, staffing roles, two years of hourly demand | **Create seed data** job |
# MAGIC | `02_train_forecast_model` | Trains an XGBoost demand model, tracks it in MLflow, registers it in Unity Catalog, and promotes `@champion` if it's better | **Train demand forecaster** job (weekly) |
# MAGIC | `03_forecast_demand` | Loads `@champion` and forecasts every venue and open hour for the coming days | **Nightly** job, task 1 |
# MAGIC | `04_optimize_staff_schedules` | Builds staff schedules with OR-Tools CP-SAT, venues solved in parallel; writes the schedules and a staffing summary | **Nightly** job, task 2 |
# MAGIC | `05_create_jobs` | Creates or updates the three jobs in your workspace (jobs as code) | Run once |
# MAGIC | `wfm_utils.py` | Shared helpers: synthetic data generator and calendar features | Imported by 01 and 03 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you start
# MAGIC You need:
# MAGIC - A Unity Catalog catalog where you can create a schema. Set it in the `catalog` widget at the top of each notebook.
# MAGIC - Permission to create personal compute
# MAGIC
# MAGIC **Get the training into your workspace as a Git folder** (skip this if you already see it):
# MAGIC 1. The repo is private: ask the repo owner to add your GitHub account as a collaborator, and accept the invite.
# MAGIC 2. Link GitHub to Databricks once: click your user icon → **Settings → Linked accounts**, choose **GitHub**, and follow
# MAGIC    the prompts to link your account.
# MAGIC 3. Go to **Workspace → your user folder → Create → Git folder**, paste `https://github.com/JoeEbbyKaruthedath/wfm-databricks-training`, and click **Create Git folder**.
# MAGIC 4. Open `00_README` inside the new folder.
# MAGIC
# MAGIC To get updates later, click the branch name at the top of any notebook in the folder and choose **Pull**.
# MAGIC
# MAGIC Every notebook has `catalog` and `schema` widgets. Changing them is all it takes to run the same code in another
# MAGIC environment (dev, pre-prod, prod).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Workspace and compute
# MAGIC **Do**
# MAGIC 1. Look around the workspace: your user folder (this Git folder lives there) and the `Shared` folder for team assets.
# MAGIC    Click the branch name at the top of this notebook: that's where you pull, commit and push.
# MAGIC 2. **Compute → Create compute**:
# MAGIC    - Policy: **Personal Compute**
# MAGIC    - Runtime: the latest **ML LTS** (it includes XGBoost and MLflow)
# MAGIC    - Terminate after **60 minutes** of inactivity
# MAGIC
# MAGIC **You should see:** the cluster running.
# MAGIC
# MAGIC **Good to know**
# MAGIC - **Personal compute** is for development. **Job compute** runs scheduled work: it's created for each run and shuts down after.
# MAGIC - Cluster policies and auto-termination keep costs under control.
# MAGIC - Notebook 04 installs the extra library it needs (OR-Tools) with `%pip install`, so there's nothing to install up front.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Unity Catalog and the seed data
# MAGIC **Do**
# MAGIC 1. Open `01_setup_and_data`, attach it to your cluster, set the `catalog` widget, and click **Run all**.
# MAGIC 2. Open **Catalog** and go to `<catalog>.wfm_training`.
# MAGIC
# MAGIC **You should see:** tables `venues`, `venue_roles` and `venue_demand_history`, and the volume `exports`.
# MAGIC
# MAGIC **Try**
# MAGIC - The **Sample data** tab on `venue_demand_history`
# MAGIC - The **Permissions** tab: one set of grants covers tables, volumes and models
# MAGIC - After steps 3–5, the **Lineage** tab: it shows which notebook or job wrote each table
# MAGIC
# MAGIC **Recreate or reset the seed data any time:** rerun `01`, or run the **WFM training: Create seed data** job.
# MAGIC Both overwrite the seed tables. The generator is deterministic, so the same settings give the same data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · MLflow: train, track and register a model
# MAGIC **Do**
# MAGIC 1. Open `02_train_forecast_model`. Set `n_estimators` = **50** and `max_depth` = **3**, then **Run all**.
# MAGIC    The first version becomes `@champion`.
# MAGIC 2. Set them back to **300** and **6** and run again.
# MAGIC
# MAGIC **You should see:** the second version has a lower holdout MAPE and is promoted to `@champion`. The first
# MAGIC version becomes `@previous_champion`.
# MAGIC
# MAGIC **Look at**
# MAGIC - **Experiments → wfm_training_demand_forecast** (in your user folder; experiments can't live inside a Git folder): select both runs and click **Compare** (parameters, metrics, the forecast plot)
# MAGIC - A run's **Inputs:** the exact table and Delta version it was trained on
# MAGIC - **Catalog → Models → venue_demand_forecaster:** versions, aliases, and the run behind each version
# MAGIC
# MAGIC **Try:** run it once more with `max_depth` = **2**. The current champion wins, so the new version is only tagged `@challenger`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Model as code: forecast with `@champion`
# MAGIC **Do:** open `03_forecast_demand` and **Run all**.
# MAGIC
# MAGIC **You should see:** "Using … (@champion)" with the version number, and a new `venue_demand_forecast` table.
# MAGIC
# MAGIC **Try**
# MAGIC 1. In **Catalog → Models → venue_demand_forecaster**, move the `@champion` alias to version 1.
# MAGIC 2. Rerun `03`. It now uses version 1, with no code change.
# MAGIC 3. Move `@champion` back to the best version.
# MAGIC
# MAGIC This is the point of shipping code, not model files: the job always asks Unity Catalog for the current champion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Spark: solving venues in parallel
# MAGIC **Do:** open `04_optimize_staff_schedules` and **Run all**.
# MAGIC
# MAGIC **You should see**
# MAGIC - `venue_staff_schedule`: the shifts for every venue, day and role
# MAGIC - `venue_staffing_summary`: required vs. scheduled staff-hours, with no understaffed hours
# MAGIC
# MAGIC **Look at:** expand **Spark Jobs** under the `applyInPandas` cell and open the stage. Its tasks share the venues between them
# MAGIC and run in parallel across the cluster's cores, and across its workers on a multi-node cluster.
# MAGIC
# MAGIC **Try:** set `service_level` = **1.2** (20% more staff) and rerun. Compare the summary.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · Jobs: create, schedule and rerun
# MAGIC **Do it by hand once**
# MAGIC 1. **Jobs & Pipelines → Create → Job**, and name it.
# MAGIC 2. Add a task `forecast_demand` that runs notebook `03`.
# MAGIC 3. Add a task `optimize_schedules` that runs `04` and **depends on** `forecast_demand`.
# MAGIC 4. For compute, pick a **job cluster**, not your personal cluster.
# MAGIC 5. Add **job parameters** `catalog` and `schema`, a **schedule** (for example, daily at 1:00 AM), and a failure **notification**.
# MAGIC 6. Click **Run now**.
# MAGIC
# MAGIC **Or create all three jobs as code:** run `05_create_jobs`. The jobs run the notebooks in this Git folder, so after a
# MAGIC **Pull** they run the latest code.
# MAGIC
# MAGIC **Try**
# MAGIC - **Run now with different parameters:** for example `service_level` = 1.1, the way a daytime rerun would work
# MAGIC - **Repair run:** if a task fails, it reruns only the failed tasks
# MAGIC - Pause and resume the schedule
# MAGIC
# MAGIC **Jobs in Git:** with the Databricks CLI, `databricks bundle generate job --existing-job-id <id>` turns a job
# MAGIC into YAML you can keep in Git, and `databricks bundle deploy -t <target>` deploys it to each environment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC - **Data, volume and model:** run `DROP SCHEMA <catalog>.wfm_training CASCADE`, or delete the schema in Catalog Explorer.
# MAGIC - **Experiment:** `/Users/<you>/wfm_training_demand_forecast`
# MAGIC - **Jobs:** delete the three "WFM training" jobs
# MAGIC - **Cluster:** terminate or delete your personal compute
