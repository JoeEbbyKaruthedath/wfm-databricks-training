# Databricks notebook source
# MAGIC %md
# MAGIC # WFM training: start here
# MAGIC
# MAGIC A small, end-to-end version of the workforce management (WFM) pipeline on Databricks, built on **synthetic data**:
# MAGIC forecast hourly guest demand per venue, then build optimized staff schedules per venue.
# MAGIC Follow along during the session, or work through the steps below on your own afterwards.
# MAGIC
# MAGIC | Notebook / file | What it does |
# MAGIC |---|---|
# MAGIC | `01_setup_and_data` | Creates the schema, the `exports` volume and the source data: 20 venues, staffing roles, two years of hourly guest counts |
# MAGIC | `02_feature_engineering` | Builds model features from the raw data, for training (`purpose = training`) or for scoring the coming days (`purpose = scoring`) |
# MAGIC | `03_train_forecast_model` | Trains an XGBoost demand model, tracks it in MLflow, registers it in Unity Catalog, and promotes `@champion` if it's better |
# MAGIC | `04_forecast_demand` | Inference: loads `@champion` and forecasts every venue and open hour in the scoring features |
# MAGIC | `05_optimize_staff_schedules` | Builds staff schedules with OR-Tools CP-SAT, venues solved in parallel; writes the schedules, hourly coverage and a summary |
# MAGIC | `wfm_utils.py` | Shared helpers: synthetic data generator and calendar features |
# MAGIC | `databricks.yml`, `resources/` | The bundle: the two pipelines, the seed-data job and the dashboard, with all notebook parameters |
# MAGIC | `dashboards/venue_staffing_plan.lvdash.json` | The **Venue staffing plan** dashboard |
# MAGIC
# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipelines: which notebook runs where
# MAGIC
# MAGIC | Pipeline (job) | Task | Notebook | Writes |
# MAGIC |---|---|---|---|
# MAGIC | Create seed data | `create_seed_data` | `01_setup_and_data` | `venues`, `venue_roles`, `venue_demand_history` |
# MAGIC | **Training:** Train demand forecaster | 1. `feature_engineering` | `02_feature_engineering` (`purpose = training`) | `venue_training_features` |
# MAGIC | | 2. `train_and_register` | `03_train_forecast_model` | model version, `@champion` / `@challenger` |
# MAGIC | **Inference:** Weekly forecast and staff schedules | 1. `feature_engineering` | `02_feature_engineering` (`purpose = scoring`) | `venue_scoring_features` |
# MAGIC | | 2. `forecast_demand` | `04_forecast_demand` | `venue_demand_forecast` |
# MAGIC | | 3. `optimize_schedules` | `05_optimize_staff_schedules` | `venue_staff_schedule`, `venue_staffing_coverage`, `venue_staffing_summary` |
# MAGIC | | 4. `refresh_dashboard` | (dashboard task) | refreshed **Venue staffing plan** |
# MAGIC
# MAGIC - **Training and inference never call each other.** Training publishes a model by moving `@champion`, and inference
# MAGIC   always loads whatever `@champion` points to.
# MAGIC - Both pipelines use the same `02_feature_engineering` notebook, so features are computed the same way for both.
# MAGIC - Both are defined in `resources/wfm_pipelines.yml` and deployed together by the bundle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you start
# MAGIC You need:
# MAGIC - A Unity Catalog catalog where you can create a schema. Set it once as `DEFAULT_CATALOG` in `wfm_utils.py`;
# MAGIC   every notebook uses it, so **Run all** works with no input. You can still override it in the `catalog` widget.
# MAGIC - Permission to create personal compute
# MAGIC
# MAGIC **Get the training into your workspace as a Git folder** (skip this if you already see it):
# MAGIC 1. Go to **Workspace → your user folder → Create → Git folder**, paste `https://github.com/JoeEbbyKaruthedath/wfm-databricks-training`, and click **Create Git folder**.
# MAGIC    The repo is public, so you don't need GitHub credentials to clone it.
# MAGIC 2. Open `00_README` inside the new folder.
# MAGIC
# MAGIC To commit and push your own changes, fork the repo and link GitHub to Databricks: user icon → **Settings → Linked accounts → GitHub**.
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
# MAGIC - **Personal compute** is for development. **Job compute** runs the pipelines: it's created for each run and shuts down after.
# MAGIC - Cluster policies and auto-termination keep costs under control.
# MAGIC - Notebook 05 installs the extra library it needs (OR-Tools) with `%pip install`, so there's nothing to install up front.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Unity Catalog and the source data
# MAGIC **Do**
# MAGIC 1. Open `01_setup_and_data`, attach it to your cluster, and click **Run all**.
# MAGIC 2. Open **Catalog** and go to `<catalog>.wfm_training`.
# MAGIC
# MAGIC **You should see:**
# MAGIC - `venues`: one row per venue (park, type, size, opening hours)
# MAGIC - `venue_roles`: the staffing rules (guests one person can serve per hour, minimum staff)
# MAGIC - `venue_demand_history`: raw guest counts per venue and hour, standing in for the data that comes from Snowflake
# MAGIC - the `exports` volume
# MAGIC
# MAGIC **Try**
# MAGIC - The **Sample data** tab on `venue_demand_history`
# MAGIC - The **Permissions** tab: one set of grants covers tables, volumes and models
# MAGIC - After the later steps, the **Lineage** tab: it shows which notebook or job wrote each table
# MAGIC
# MAGIC **Recreate or reset the source data any time:** rerun `01`, or run the **WFM training: Create seed data** job.
# MAGIC Both overwrite the seed tables. The generator is deterministic, so the same settings give the same data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Feature engineering
# MAGIC **Do**
# MAGIC 1. Open `02_feature_engineering`, set `purpose` = **training**, and click **Run all**.
# MAGIC 2. Set `purpose` = **scoring** and run it again.
# MAGIC
# MAGIC **You should see:**
# MAGIC - `venue_training_features`: every venue-hour of the history, with the features and the actual guests (the label)
# MAGIC - `venue_scoring_features`: every venue-hour of the next 7 days, with the same features and no label
# MAGIC
# MAGIC **Why one notebook for both:** training and scoring compute features with the same code, so the model never sees
# MAGIC features built two different ways.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · MLflow: train, track and register a model
# MAGIC **Do**
# MAGIC 1. Open `03_train_forecast_model`. Set `n_estimators` = **50** and `max_depth` = **3**, then **Run all**.
# MAGIC    The first version becomes `@champion`.
# MAGIC 2. Set them back to **300** and **6** and run again.
# MAGIC
# MAGIC **You should see:** the second version has a much lower holdout MAPE and is promoted to `@champion`. The first
# MAGIC version becomes `@previous_champion`.
# MAGIC
# MAGIC **Look at**
# MAGIC - **Experiments → wfm_training_demand_forecast** (in your user folder; experiments can't live inside a Git folder): select both runs and click **Compare** (parameters, metrics, the forecast plot)
# MAGIC - A run's **Inputs:** the exact feature table and Delta version it was trained on
# MAGIC - **Catalog → Models → venue_demand_forecaster:** versions, aliases, and the run behind each version
# MAGIC
# MAGIC **Try:** run it once more with `max_depth` = **2**. The current champion wins, so the new version is only tagged `@challenger`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Model as code: inference with `@champion`
# MAGIC **Do:** open `04_forecast_demand` and **Run all**.
# MAGIC
# MAGIC **You should see:** "Using … (@champion)" with the version number, and a new `venue_demand_forecast` table.
# MAGIC
# MAGIC **Try**
# MAGIC 1. In **Catalog → Models → venue_demand_forecaster**, move the `@champion` alias to version 1.
# MAGIC 2. Rerun `04`. It now uses version 1, with no code change.
# MAGIC 3. Move `@champion` back to the best version.
# MAGIC
# MAGIC This is the point of shipping code, not model files: the pipeline always asks Unity Catalog for the current champion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · Spark: solving venues in parallel
# MAGIC **Do:** open `05_optimize_staff_schedules` and **Run all**.
# MAGIC
# MAGIC **You should see**
# MAGIC - `venue_staff_schedule`: the shifts for every venue, day and role
# MAGIC - `venue_staffing_coverage`: staff needed vs. on duty for every venue, role and hour (the dashboard uses it)
# MAGIC - `venue_staffing_summary`: needed vs. scheduled staff-hours per venue, with no understaffed hours
# MAGIC
# MAGIC **Look at:** expand **Spark Jobs** under the `applyInPandas` cell and open the stage. Its tasks share the venues between them
# MAGIC and run in parallel across the cluster's cores, and across its workers on a multi-node cluster.
# MAGIC
# MAGIC **Try:** set `service_level` = **1.2** (20% more staff) and rerun. Compare the summary.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 · Deploy and run the pipelines with the bundle
# MAGIC The bundle (`databricks.yml` plus `resources/`) defines the two pipelines, the seed-data job and the dashboard as code,
# MAGIC including every notebook parameter.
# MAGIC
# MAGIC **Deploy**
# MAGIC 1. Open `databricks.yml` and set your catalog: uncomment `variables:` and `catalog:` under the `dev` target.
# MAGIC 2. **From the workspace:** with `databricks.yml` open, click the **Deployments** icon (rocket), choose the `dev` target,
# MAGIC    and click **Deploy**. This needs serverless compute enabled in the workspace.
# MAGIC 3. **Or from your laptop**, with the Databricks CLI:
# MAGIC    `databricks bundle deploy -t dev --var="catalog=<your_catalog>"`
# MAGIC
# MAGIC The `dev` target prefixes the job names with `[dev <your name>]` and pauses every schedule. The `prod` target keeps the
# MAGIC weekly inference schedule running.
# MAGIC
# MAGIC **Run:** open **Jobs & Pipelines**, or use the run (play) button under **Bundle resources**, in this order:
# MAGIC 1. **Create seed data**
# MAGIC 2. **Train demand forecaster**
# MAGIC 3. **Weekly forecast and staff schedules**
# MAGIC
# MAGIC **Look at:** each job's **Tasks** graph, **Parameters**, **Schedules & Triggers** and **Runs**.
# MAGIC
# MAGIC **Try**
# MAGIC - **Run now with different parameters:** for example `service_level` = 1.1, the way a daytime rerun would work
# MAGIC - **Repair run:** if a task fails, it reruns only the failed tasks
# MAGIC - Change a parameter default in `resources/wfm_pipelines.yml`, deploy again, and see the job update
# MAGIC
# MAGIC **Good to know:** to build a job by hand, use **Jobs & Pipelines → Create → Job**. Then
# MAGIC `databricks bundle generate job --existing-job-id <id>` turns it into bundle YAML.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 · See the results in the dashboard
# MAGIC **Do:** open **Dashboards → Venue staffing plan**. The bundle deploys it, and the last task of the inference pipeline
# MAGIC refreshes it.
# MAGIC
# MAGIC **You should see two pages**
# MAGIC - **Staffing plan:**
# MAGIC   - shifts, staff-hours, staffing buffer and hours below need
# MAGIC   - staff on duty vs. needed by hour, staff-hours by day and role, and by venue
# MAGIC   - the shift schedule
# MAGIC - **Guest demand:**
# MAGIC   - forecast guests, compared with the last 8 weeks
# MAGIC   - guests per day, when guests arrive by day and hour, and the busiest venues
# MAGIC
# MAGIC **Try**
# MAGIC - Filter to one park or venue type in the filter panel, or pick a single day on the Staffing plan page.
# MAGIC - Run the inference pipeline with `service_level` = 1.2. When it finishes, the staffing buffer on the dashboard is higher.
# MAGIC
# MAGIC **Good to know:** dashboards run on a SQL warehouse (the bundle's `warehouse_id` variable).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC - **Jobs and dashboard:** `databricks bundle destroy -t <target>`, or delete them in the UI
# MAGIC - **Data, volume and model:** run `DROP SCHEMA <catalog>.wfm_training CASCADE`, or delete the schema in Catalog Explorer
# MAGIC - **Experiment:** `/Users/<you>/wfm_training_demand_forecast`
# MAGIC - **Cluster:** terminate or delete your personal compute
