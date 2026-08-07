# Conflict Escalation Forecasting — Project Guide

### ML Bubble 2026 (AIT Pune) — Defense & National Security Track

---

## 1\. Problem Statement

Most student "defense ML" projects stop at classifying conflict events after the fact (e.g., "was this a battle or a protest?"). This project goes a step further: **predict where and when conflict activity is likely to escalate in the next 7–30 days**, using historical event patterns as a spatiotemporal signal.

Framed as a forecasting problem (not a one-shot classification problem), this demonstrates a more mature understanding of the domain — which is exactly what "advanced" tracks (SE / TE-BE) are judged on.

**One-line pitch for your PPT:** *"A district-level early-warning model that forecasts conflict escalation risk over a rolling 7–30 day window, using engineered temporal features from ACLED event data."*

---

## 2\. Where to Get the Dataset

### Primary source: ACLED (Armed Conflict Location & Event Data Project)

ACLED provides free real-time data on political violence, conflict events, and protests globally, with location and date information for each event. Access requires registration:

- **Register:** [https://acleddata.com](https://acleddata.com) — create a free "myACLED" account  
- **Data Export Tool** (easiest for a hackathon): [Data Export Tool | ACLED](https://acleddata.com/conflict-data/data-export-tool) — lets you filter and download a specific subset of the data by location, event type, and date range  
- **API access:** available after registration for programmatic pulls. Note that API responses default to 5,000 rows per call, so for larger date ranges you'll need to filter and paginate your requests

**What the data contains (per event):** event date, event type (battle, explosion/remote violence, violence against civilians, protest, riot, strategic development), actor types, fatalities, country/region, admin1–admin3 location, and precise latitude/longitude.

**Recommended scope for your project (to keep it tractable in the time you have):**

- Pick 1–2 countries/regions with a reasonably active and well-documented conflict history (South Asia, Sahel, or Horn of Africa region tend to have dense, well-labeled data)  
- Pull 3–5 years of event history — enough to build meaningful rolling-window features without becoming unmanageable

### Backup / supplementary sources

- **HDX (Humanitarian Data Exchange) ACLED mirror:** HDX hosts weekly aggregated ACLED datasets by country, with civilian-targeting and fatality counts already summarized — useful if the full API feels like too much setup time. [https://data.humdata.org/organization/acled](https://data.humdata.org/organization/acled)  
- **UCDP GED (Uppsala Conflict Data Program):** an alternative/complementary event dataset used heavily in academic conflict-forecasting work — good for cross-validating findings or enriching features. [https://ucdp.uu.se/downloads/](https://ucdp.uu.se/downloads/)  
- **GDELT Project:** global news-event data, useful if you want to add a news-based "narrative tension" feature. [https://www.gdeltproject.org/](https://www.gdeltproject.org/)

---

## 3\. Reference GitHub Repositories

You should **not copy these directly** (judges will check originality), but they're excellent for understanding methodology, feature engineering patterns, and what a credible pipeline looks like.

1. **ACLED's own official forecasting pipeline (CAST):** [https://github.com/ACLED/cast-public](https://github.com/ACLED/cast-public) This is ACLED's Conflict Alert System, which forecasts organized political violence up to six months ahead at global, national, and sub-national levels, disaggregated by event type, built around a hierarchical LightGBM model with uncertainty estimation. This is the gold-standard reference for how a real org structures this problem — worth skimming their `feature_engineer.py` approach (lags, rolling counts, counts-since-last-event) even if you don't use their code.  
     
2. **Explainable conflict fatality forecasting:** [https://github.com/fif911/probabilistic\_conflict\_modelling](https://github.com/fif911/probabilistic_conflict_modelling) A publicly available, explainable early-conflict-forecasting model that predicts the distribution of conflict-related fatalities at the country-month level, built on UCDP GED, V-Dem, World Development Indicators, and ACLED data, producing predictions up to 14 months ahead. Good reference for framing this as a probabilistic/regression problem rather than pure classification — a nice "advanced" talking point if a judge asks about uncertainty.  
     
3. **Harvard Kennedy School civil conflict onset project:** found via GitHub Topics → `conflict-prediction`: [https://github.com/topics/conflict-prediction](https://github.com/topics/conflict-prediction) Includes a project built for a Machine Learning and Data Analytics course that created a novel dataset to explore how ML can predict the onset of civil conflict — good for feature-engineering inspiration on the "onset" side (adjacent to escalation).  
     
4. **General ACLED tooling (for data pulls):** [https://github.com/chris-dworschak/acled.api](https://github.com/chris-dworschak/acled.api) — an R package that wraps the ACLED API to make retrieving conflict event data straightforward (use if any teammate is comfortable in R; otherwise just hit the REST API directly in Python with `requests`).

Browse the full topic page for more (Myanmar spatiotemporal prediction, GDELT-based geopolitical risk dashboards, etc.): [https://github.com/topics/conflict-prediction](https://github.com/topics/conflict-prediction)

---

## 4\. Methodology / Model Approach

### Step 1 — Define the prediction target

Pick ONE clear framing (don't try to do all of these):

- **Binary classification:** will admin1/admin2 region X see a conflict escalation (e.g., fatality count or event count rising above a threshold) in the next 7/14/30 days? — *simplest, best if time is short*  
- **Regression:** predict number of conflict events or fatalities in the next window  
- **Risk score (0–1):** probability of escalation — nicer for a dashboard demo

**Recommendation given your deadline: binary classification with a 14-day horizon.** It's fast to build, easy to explain, and still "advanced" enough with good features.

### Step 2 — Feature engineering (this is where the project earns its "advanced" label)

Build rolling-window features per region, computed as of each date:

- Event count in last 7 / 14 / 30 days  
- Fatality count in last 7 / 14 / 30 days  
- Event type diversity (Shannon entropy across event types)  
- Actor diversity (number of distinct armed actors active)  
- "Escalation velocity" — rate of change between this window and the previous window (e.g., events\_last\_7d − events\_prior\_7d)  
- Days since last event in the region  
- Rolling mean/std of fatalities (volatility signal)  
- Neighboring-region spillover (optional, more advanced): aggregate event counts in geographically adjacent admin regions

### Step 3 — Model

- **XGBoost or LightGBM** on the tabular rolling-window features — matches ACLED's own production approach (their CAST system is built around a hierarchical LGBM model), which is a strong credibility point to mention in your presentation.  
- Use **time-based train/test split** (never random split — conflict data is a time series, and random splitting leaks future information into training).  
- Report **precision/recall/F1**, not just accuracy — conflict escalation is a rare/imbalanced event, so accuracy alone will look artificially high.

### Step 4 — Explainability (important for the "model explanation round")

- Use **SHAP values** to show which features drove a specific prediction (e.g., "rising actor diversity \+ fatality volatility flagged this district")  
- This directly prepares you for the Aug 16 online round where you must explain your methodology.

### Step 5 (stretch goal, if time allows) — Simple map visualization

A choropleth or scatter map (Plotly / Folium) showing predicted risk by region for the next window makes for a strong demo visual — judges remember what they can *see*.

---

## 5\. Suggested Repo Structure

conflict-escalation-forecasting/

├── data/

│   ├── raw/                  \# ACLED CSV exports (don't commit if large — .gitignore)

│   └── processed/            \# feature-engineered datasets

├── notebooks/

│   ├── 01\_eda.ipynb

│   ├── 02\_feature\_engineering.ipynb

│   └── 03\_modeling.ipynb

├── src/

│   ├── data\_loader.py        \# ACLED API / CSV ingestion

│   ├── feature\_engineer.py   \# rolling window features

│   ├── train.py

│   └── evaluate.py

├── models/

│   └── xgb\_escalation\_model.pkl

├── reports/

│   ├── model\_metrics.md

│   └── shap\_summary.png

├── README.md

└── requirements.txt

A clean README with problem statement, dataset source/citation, methodology, and metrics is graded — don't skip it.

---

## 6\. Suggested Timeline (given your deadline)

| Time | Task |
| :---- | :---- |
| Today | Register on ACLED, pull data for 1–2 countries (3–5 yrs), set up repo skeleton |
| Today/Tomorrow AM | EDA \+ feature engineering |
| Tomorrow midday | Train XGBoost baseline, time-based split, get F1/precision/recall |
| Tomorrow afternoon | SHAP explainability \+ basic map/plot visualization |
| Tomorrow evening | PPT/PDF, README, GitHub repo cleanup, submit before 10 PM IST |

---

## 7\. Key Academic Reference (for credibility in your write-up)

Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). *Introducing ACLED: An Armed Conflict Location and Event Dataset.* Journal of Peace Research, 47(5), 651–660. — This is the original ACLED paper; citing it in your documentation shows academic rigor.

---

*Note: ACLED data is free but requires attribution per their terms of use when you publish or submit results — include an attribution line in your README and PPT (e.g., "Data: ACLED, acleddata.com").*  
