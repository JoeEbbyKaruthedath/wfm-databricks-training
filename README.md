# WFM training on Databricks

Hands-on training material: a small, end-to-end workforce management (WFM) pipeline on Databricks, built on synthetic
data. It forecasts hourly guest demand for each venue, then builds optimized staff schedules for each venue.

| Notebook | What it does |
|---|---|
| `00_README` | Start here: the step-by-step lab guide |
| `01_setup_and_data` | Creates the schema, the `exports` volume and the seed data: 20 venues, staffing roles, two years of hourly demand |
| `02_train_forecast_model` | Trains an XGBoost demand model, tracks it in MLflow, registers it in Unity Catalog, and promotes `@champion` if it beats the current champion |
| `03_forecast_demand` | Loads `@champion` and forecasts every venue and open hour for the coming days |
| `04_optimize_staff_schedules` | Builds staff schedules with OR-Tools CP-SAT, solving venues in parallel with Spark |
| `05_create_jobs` | Creates or updates three jobs: seed data, weekly training, nightly forecast and schedules |
| `wfm_utils.py` | Shared helpers: synthetic data generator and calendar features |

## Get started

1. This repo is private. Ask the owner to add your GitHub account as a collaborator, and accept the invite.
2. In Databricks, link GitHub once: user icon → **Settings → Linked accounts** → **GitHub**.
3. Go to **Workspace → your user folder → Create → Git folder**, paste this repo's URL, and create the Git folder.
4. Open `00_README` and follow the steps.

You need a Unity Catalog catalog where you can create a schema, and permission to create personal compute.
The notebooks run on Databricks Runtime ML (LTS).

## What it creates

- **Schema** `<catalog>.wfm_training`: tables `venues`, `venue_roles`, `venue_demand_history`, `venue_demand_forecast`,
  `venue_staff_schedule`, `venue_staffing_summary`; volume `exports`; model `venue_demand_forecaster`
- **MLflow experiment** `/Users/<you>/wfm_training_demand_forecast`
- **Jobs** (from `05_create_jobs`): *WFM training: Create seed data*, *WFM training: Train demand forecaster*,
  *WFM training: Nightly forecast and staff schedules*. Schedules start paused.

All data is synthetic.
