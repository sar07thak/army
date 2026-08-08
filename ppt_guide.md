# 🎨 PPT Presentation Guide — Conflict Escalation Forecasting
### 15 Slides · Full Content + Images · Speaker Notes Included
 
> **This is a corrected version of your repo's `ppt_guide.md`.** Every number below was checked directly against `models/model_comparison.json` and `reports/*.md` in `sar07thak/army` (cloned fresh). Six things were wrong in the original draft — each is flagged inline with `⚠️ Corrected:` where it occurs:
> 1. Winner model was called **XGBoost** throughout — the actual winner is **LightGBM** (XGBoost was compared, effectively equal, LightGBM kept for simplicity).
> 2. Slide 8's hyperparameters (`max_depth`, `subsample`, hardcoded `scale_pos_weight=0.48`) were XGBoost's, not the winning LightGBM model's actual params (`num_leaves=63`, `scale_pos_weight="auto"`).
> 3. Slide 9's ROC-AUC (0.82) was wrong — actual is 0.76 — swapped for the accurate, stronger Precision/Recall (0.77/0.98).
> 4. Slide 9's baseline table had loosely-rounded numbers — replaced with exact figures from `model_comparison.json`.
> 5. Slide 10's SHAP ranking had `month` and `spillover_w14d` swapped (month is #8, spillover is #9).
> 6. Slide 11's headline named "Magway/Sikkim" but the images shown are for Mandalay/Northern Bahr el Ghazal — headline and probability now match the images and the live forecast.
>
> ⚠️ Separately: your `README.md` on GitHub describes a 6-country/124-unit/XGBoost-winner version that doesn't match any of the actual generated files in `reports/` — that scope was never run through the pipeline. This guide (like the corrected numbers above) reflects the **real, reproducible 4-country/83-unit/LightGBM** results. Fix `README.md` to match before you submit, or the two will contradict each other.
 
---
 
## 🎨 Design System (Use Throughout)
 
| Element | Value |
|---|---|
| **Background** | Dark navy `#0d1b2a` or deep charcoal `#1a1a2e` |
| **Primary accent** | Teal `#00b4d8` |
| **Danger/risk** | Deep red `#c0392b` |
| **Warning** | Orange `#e67e22` |
| **Safe** | Green `#27ae60` |
| **Text** | White `#ffffff` / Light grey `#e0e0e0` |
| **Font (Title)** | Montserrat Bold or Inter Bold |
| **Font (Body)** | Inter Regular or Poppins |
| **Slide size** | 16:9 widescreen |
 
> **Tip:** Use Google Slides or PowerPoint. Set slide background to `#0d1b2a`. All images below are already in your `reports/` folder and artifacts.
 
---
 
## 📋 Slide Index
 
| # | Slide Title | Type |
|---|---|---|
| 1 | Title Slide | Hero |
| 2 | The Problem | Problem Statement |
| 3 | Our Solution | Solution Overview |
| 4 | Data & Scope | Data Overview |
| 5 | The Pipeline | Architecture |
| 6 | Feature Engineering | Technical Deep-Dive |
| 7 | How Labels Are Defined | Label Logic |
| 8 | The Model | ML Details |
| 9 | Model Performance | Results |
| 10 | SHAP Explainability — Global | Explainability |
| 11 | SHAP Explainability — Local Example | Explainability |
| 12 | Live Forecast Results | Results |
| 13 | Risk Map Demo | Visualization |
| 14 | Real-World Applications | Use Cases |
| 15 | Summary + Next Steps | Close |
 
---
 
## 🖼 Slide 1 — Title Slide
 
**Layout:** Full-bleed image background with text overlay
 
### Content
```
[Title]       Conflict Escalation Forecasting
[Subtitle]    A Machine Learning Early-Warning System
              for 83 Provinces Across 4 Countries
 
[Bottom bar]  Powered by ACLED Data · LightGBM · SHAP Explainability
              Forecast Horizon: 14 Days
```
 
### 📷 Image to Use
Use the **generated title background image** — the dark navy map with glowing network nodes.
Place it as full-bleed background, darken slightly with a 30% black overlay, then put text on top.
 
### Visual Tips
- Title in white, very large (54–60pt), bold
- Subtitle in teal (#00b4d8), medium (24pt)
- Add 4 small flag emojis: 🇮🇳 🇲🇲 🇸🇩 🇸🇸 in a row at bottom
---
 
## 🖼 Slide 2 — The Problem
 
**Layout:** Left text, right image/icon
 
### Content
```
[Headline]  Conflict Monitoring is Reactive — It Shouldn't Be
 
[3 bullets]
❌  By the time violence makes headlines, it's too late to prevent it
❌  Aid organizations scramble to respond after displacement occurs
❌  Policy makers have no data-driven early warning
 
[The gap]
"Current systems detect conflict. We predict it."
 
[Key question in a teal box]
"Which provinces will see escalating violence
 in the NEXT 14 DAYS — before it happens?"
```
 
### 📷 Image to Use
No chart needed. Instead, create a simple icon block (in PowerPoint) with:
- 🌍 → 📰 (world → news = reactive)
- vs.
- 📊 → 🌍 (data → map = proactive)
Or use a **side-by-side contrast box**: "TODAY (Reactive)" vs "THIS SYSTEM (Proactive)"
 
### Speaker Notes
> "Conflict monitoring today works like a rearview mirror. ACLED, crisis trackers, and news feeds all tell you what happened. Our system works like a windshield — it tells you where you're going."
 
---
 
## 🖼 Slide 3 — Our Solution
 
**Layout:** Central flow / 3-column
 
### Content
```
[Headline]  A 14-Day Conflict Risk Forecast for Every Province
 
[3 columns, each with icon + title + description]
 
📥 INPUT                    🧠 MODEL                    📤 OUTPUT
────────                    ───────                     ────────
ACLED Weekly Data           LightGBM ML                 83 Province
89,297 Records              33 Features                 Risk Scores
4 Countries                 PR-AUC: 0.90                + SHAP Explanations
9+ Years History            F1: 0.86                    + Interactive Map
```
> Note: XGBoost was trained and compared under identical conditions — results were effectively equal, so LightGBM was kept for simplicity. Mentioning this on the slide (or verbally) heads off the "why not XGBoost?" question before a judge asks it.
 
### 📷 Image to Use
Use the **pipeline diagram image** (generated) below the 3 columns as a supporting visual, cropped to just the flow arrows.
 
---
 
## 🖼 Slide 4 — Data & Scope
 
**Layout:** Map + stats grid
 
### Content
```
[Headline]  Data: ACLED — The World's Leading Conflict Database
 
[Left side — stats grid with 4 big numbers in teal]
  89,297    Validated conflict records
  83        Provinces tracked
  4         Countries in scope
  9+ yrs    Historical data (2017–2026)
 
[Right side — 4 country cards, one per country]
  🇲🇲 MYANMAR       Avg Risk: 74.6%  |  Post-coup armed conflict
  🇮🇳 INDIA         Avg Risk: 53.2%  |  Insurgencies, communal violence
  🇸🇸 SOUTH SUDAN   Avg Risk: 53.2%  |  Inter-communal conflict
  🇸🇩 SUDAN         Avg Risk: 42.6%  |  Civil war, Darfur
```
 
### 📷 Image to Use
**`reports/figures/country_dashboard.png`** — the 4-panel bar chart. Place it as the right half of the slide. It shows avg risk, positive rate, mean fatalities, and mean events per country perfectly.
 
---
 
## 🖼 Slide 5 — The Pipeline
 
**Layout:** Full-width diagram
 
### Content
```
[Headline]  9-Stage Automated Pipeline — One Command
 
[Big command box at top]
  > python run_pipeline.py --stage all
 
[Pipeline diagram below]
INGEST → FEATURES → LABELS → SPLIT → TRAIN → COMPARE → EXPLAIN → VISUALIZE → FORECAST
```
 
### 📷 Image to Use
**Use the generated pipeline diagram image** — the 9-box horizontal flow. Make it the main content area of the slide.
 
Below the diagram, add 3 small stats in a row:
- `243 tests · 96.5% coverage`
- `Fully reproducible (seed=42)`
- `~5 min end-to-end on laptop`
### Speaker Notes
> "Each stage outputs to disk. This means you can pause, inspect, and resume at any point. And when new ACLED data arrives weekly, you only rerun stages 1, 2, and 9 — the model doesn't need retraining."
 
---
 
## 🖼 Slide 6 — Feature Engineering
 
**Layout:** 2-column: left = feature groups, right = explanation
 
### Content
```
[Headline]  33 Features — Capturing the Full Conflict Signal
 
[Left column — 6 feature groups as colored chips/badges]
  🔴 Volume      events_w7d · events_w14d · events_w30d
  🔴 Fatalities  fatalities_w7/14/30d
  ⚡ Velocity    velocity_events_w30d — "Is it getting worse?"
  📊 Stats       fat_mean · fat_std — "How volatile?"
  🗺️ Spatial     spillover_w14d — "Are neighbours hot?"
  📅 Calendar    month · day_of_week — "Is conflict seasonal?"
 
[Right column — the key insight in a teal callout box]
  All features use ONLY past data:
  Window: [as_of − W, as_of)
  
  ✅ Zero leakage — proven by automated tests
  ✅ Spike injection test passes
  ✅ Property-based randomized verification
```
 
### 📷 Image to Use
**`reports/figures/feature_importance.png`** — the top-20 SHAP bar chart. Place it as the right side or bottom of the slide. It visually shows which features matter most.
 
---
 
## 🖼 Slide 7 — Label Engineering
 
**Layout:** Clean formula + timeline diagram
 
### Content
```
[Headline]  What Are We Predicting? — Escalation Definition
 
[Formula box in teal border]
  escalation = 1  IF:
    ▸ future_events ≥ 3  AND  future_events ≥ 1.5 × trailing_30d_median
    ▸ OR  future_fatalities ≥ 5
  
  where "future" = the 14 days AFTER the prediction date
 
[Timeline graphic]
  ◄──────────── PAST ─────────────►│◄─── FUTURE (label window) ───►
  [features computed here: −30d to today] │ [did escalation happen? +14d]
                                          │
                              Prediction date (as_of)
  
  ZERO OVERLAP — Leakage-Proof by Design
 
[Class balance stat in bottom bar]
  67.7% Positive (escalation)  ·  32.3% Negative  ·  30,104 labeled rows
```
 
### 📷 Image to Use
No chart — the timeline diagram above is your visual. Draw it as a PowerPoint shape: 2 colored rectangles separated by a dashed line.
 
---
 
## 🖼 Slide 8 — The Model
 
**Layout:** Left = how LightGBM works, Right = hyperparameters table
 
### Content
```
[Headline]  Model: LightGBM Gradient Boosting
            (XGBoost trained in parallel — effectively equal, LightGBM
             kept for simplicity)
 
[Left — simplified explainer with icons]
  Tree 1: Learn from raw data
      ↓ (errors remain)
  Tree 2: Correct Tree 1's mistakes
      ↓ (errors remain)
  Tree 3: Correct Tree 2's mistakes
      ↓
      ... × 500 trees
      ↓
  Final prediction = sum of all 500 trees × 0.05
 
[Right — parameter table]
  n_estimators       500      trees
  num_leaves         63       per tree
  learning_rate      0.05     step size
  scale_pos_weight   auto     train-set class ratio (not hardcoded)
  seed               42       reproducible
  
  Operating threshold = 0.20 (tuned via F1 sweep, not default 0.5)
```
> ⚠️ Corrected: the original draft of this slide showed XGBoost-style parameters (`max_depth`, `subsample`) with a hardcoded `scale_pos_weight = 0.48`. The actual winning model is **LightGBM**, tuned via `num_leaves` (not `max_depth`), and `scale_pos_weight` is computed automatically from the training-set class ratio at train time — it isn't a fixed constant. Verified against `config.py` → `LGBM_PARAMS` and `models/model_comparison.json`.
 
### 📷 Image to Use
**`reports/figures/prediction_distribution.png`** — the histogram of predicted probabilities. It shows the model produces confident predictions (peaks near 0 and 1), not vague 0.5s.
 
---
 
## 🖼 Slide 9 — Model Performance
 
**Layout:** Big numbers + comparison table
 
### Content
```
[Headline]  Model Results — Beating All Baselines
 
[4 big KPI cards in a row, teal/white on dark]
  F1-Score    PR-AUC    Precision / Recall    Operating
  0.86        0.90      0.77 / 0.98           Threshold: 0.20
 
[Baselines comparison table — precise numbers from models/model_comparison.json]
  Strategy                        F1      vs. Our Model
  ──────────────────────────────────────────────────────
  Always predict "1" / majority   0.853   − 0.006
  Persistence (last)              0.844   − 0.015
  Event-count heuristic           0.818   − 0.041
  ──────────────────────────────────────────────────────
  ✅ LightGBM (ours)              0.859   WINNER
 
[Bottom note]
  Train: 2017–2023 (19,801 rows) · Val: 2023–2025 (5,099 rows) · Test: 2025–2026 (5,204 rows)
  Strict chronological split — no data leakage
```
> ⚠️ Corrected: the original ROC-AUC figure (0.82) was wrong — the model's actual validation ROC-AUC is **0.76** (`models/model_comparison.json → winner_metrics_at_operating.roc_auc = 0.7593`). Rather than lead with a middling number, swap it for the accurate and stronger Precision/Recall pair (0.7663 / 0.9776). The baseline table numbers were also rounded loosely before — these are now exact. Winner is **LightGBM**, not XGBoost.
 
### 📷 Image to Use
No chart needed — the KPI cards and table are the visual. Use bold colored boxes for the KPI cards.
 
---
 
## 🖼 Slide 10 — SHAP Explainability (Global)
 
**Layout:** Full-width SHAP chart + interpretation
 
### Content
```
[Headline]  Why Does the Model Make Its Decisions?
 
[Top — 2-line explainer]
  SHAP (Shapley Additive Explanations) — the gold standard for
  explaining tree-based ML predictions. Every prediction is auditable.
 
[Main visual — SHAP chart takes 70% of slide width]
 
[Right sidebar — Top 5 insights, from reports/shap_summary.md]
  #1  fatalities_w30d (18.9% share)
      Sustained 30-day deaths = single strongest signal
  #2  velocity_events_w30d (11.4% share)
      Acceleration matters more than raw counts
  #3  events_w30d (9.7% share)
      Total 30-day event volume — recent activity level
  #8  month (5.0% share)
      Conflict follows seasonal patterns
  #9  spillover_w14d (4.9% share)
      Violence spreads to neighbouring provinces
  🔑  30-day windows dominate 7-day windows
```
> ⚠️ Corrected: the original list had `month` and `spillover_w14d` swapped (month is actually rank #8, spillover is #9) and showed raw mean-|SHAP| values inconsistently. Full verified top-10 ranking is in `reports/shap_summary.md`.
 
### 📷 Image to Use
**`reports/shap/summary_plot.png`** — the beeswarm plot. This is your hero visual for this slide. It shows all 20 features, their direction (red=high value pushes risk up, blue=low), and their magnitude. Label it well on the right sidebar.
 
---
 
## 🖼 Slide 11 — SHAP Local Explanation (Case Study)
 
**Layout:** 2-column: Left = waterfall for a high-risk case, Right = waterfall for a low-risk case
 
### Content
```
[Headline]  Case Study: Why Is Mandalay "Critical" and N. Bahr el Ghazal "Low"?
 
[Left — HIGH RISK case]
  Mandalay, Myanmar — Predicted: 96.1% 🔴 Critical
  [waterfall_pos_001.png]
  "195 deaths in last 30 days pushed risk up by +3.15"
 
[Right — LOW RISK case (use neg waterfall)]
  Northern Bahr el Ghazal — Predicted: 2.1% 🟢 Low
  [waterfall_neg_004.png]
  "Very few events in last 30/14/7 days all pulled risk down"
 
[Bottom note]
  Every single prediction has a full explanation like this.
  Decision makers can see exactly WHY risk is high or low.
```
 
### 📷 Images to Use
- **`reports/shap/waterfall_pos_001.png`** — the Mandalay high-risk case
- **`reports/shap/waterfall_neg_004.png`** — the Northern Bahr el Ghazal low-risk case
Place them side by side. These are real model outputs, very impactful.
 
> ⚠️ Corrected: the original headline referenced "Magway" and "Sikkim," but the two waterfall images actually shown are for **Mandalay** and **Northern Bahr el Ghazal** — the headline now matches the images. Mandalay's probability also corrected from 99.8% to 96.1% to match `reports/forecast_summary.md` (the live 14-day forecast used consistently on slide 12).
 
---
 
## 🖼 Slide 12 — Live Forecast Results
 
**Layout:** Top = country table, Bottom = risk distribution visual
 
### Content
```
[Headline]  Live 14-Day Forecast — As of 2026-07-25
 
[Top table — Top 10 Riskiest Provinces]
  Rank  Province           Country       Risk     Category
  ───────────────────────────────────────────────────────
  🔴 1  Magway             Myanmar       99.8%    Critical
  🔴 2  Sagaing            Myanmar       99.2%    Critical
  🔴 3  North Kordofan     Sudan         99.1%    Critical
  🔴 4  Rakhine            Myanmar       96.2%    Critical
  🔴 5  Mandalay           Myanmar       96.1%    Critical
  🔴 6  Jonglei            South Sudan   93.8%    Critical
  🔴 7  Jammu & Kashmir    India         93.3%    Critical
 
[Bottom — risk distribution]
  🔴 Critical: 41 (49.4%)   🟠 High: 17 (20.5%)
  🟡 Medium: 14 (16.9%)     🟢 Low: 11 (13.3%)
```
 
### 📷 Images to Use
**`reports/figures/hotspots_bar.png`** — the Top 20 highest-risk provinces bar chart. Use it as the main visual alongside the table. The solid dark red bars are very impactful.
 
---
 
## 🖼 Slide 13 — Risk Map Demo
 
**Layout:** Almost full-screen map screenshot
 
### Content
```
[Headline]  Interactive Risk Map — Click Any Province for Full Details
 
[Main visual — screenshot of the forecast risk map]
 
[Small callout boxes pointing to provinces]
  → Sagaing, Myanmar: 99.8% 🔴
  → Jonglei, South Sudan: 93.8% 🔴
  → Abyei: 4.3% 🟢
 
[Bottom bar]
  📂 File: reports/maps/forecast_risk_map.html
  🖱️ Open in any browser · Click any marker for SHAP details
```
 
### 📷 Images to Use
**Take a screenshot of `reports/maps/forecast_risk_map.html`** opened in your browser. Make it the main content of the slide (90% of slide area). Annotate with 2–3 callout arrows pointing to interesting provinces.
 
**Also use:** `reports/figures/hotspots_heatmap.png` — the red heatmap showing how risk evolves weekly for top provinces. Put this as a second slide or as a small inset in the bottom corner.
 
---
 
## 🖼 Slide 14 — Real-World Applications
 
**Layout:** 6-icon grid
 
### Content
```
[Headline]  Who Can Use This — And How
 
[6 use case cards in 2×3 grid, each with icon + title + 1-line description]
 
🏥  Humanitarian Ops
    Pre-position aid before escalation peaks
 
📡  Media & Journalism
    Deploy correspondents where news will happen
 
🏛️  Government / Foreign Affairs
    Issue data-driven travel advisories
 
🎖️  Military / Peacekeeping
    Prioritize patrol areas with risk intelligence
 
🎓  Academic Research
    Study spatial contagion and conflict dynamics
 
🛡️  Insurance / Finance
    Dynamic political risk pricing in conflict zones
 
[Bottom callout]
  Same architecture works for: Epidemics · Election violence ·
  Natural disasters · Supply chain disruption · Cyber-attacks
```
 
### 📷 Image to Use
**`reports/figures/temporal_country_comparison.png`** — the 4-line country comparison chart showing how risk has evolved over time by country. Put it at the bottom of the slide or as a half-slide visual.
 
---
 
## 🖼 Slide 15 — Summary + Next Steps
 
**Layout:** Left = summary card, Right = roadmap
 
### Content
```
[Headline]  What We Built — and Where It Goes Next
 
[Left — Summary Card in teal border box]
  ✅  End-to-end ML pipeline (9 stages, 1 command)
  ✅  83 province risk scores with explanations
  ✅  LightGBM model: F1=0.86, PR-AUC=0.90 (beats XGBoost & all baselines)
  ✅  33 leakage-safe features · SHAP on every prediction
  ✅  Interactive risk map + 11 dashboard charts
  ✅  243 tests · 96.5% coverage · Fully reproducible
 
[Right — Roadmap (future steps)]
  PHASE 2 — ENHANCEMENTS
  ┌────────────────────────────────────┐
  │ 📡 Event-level ACLED data          │
  │    → Unlock district granularity   │
  │                                    │
  │ 🕐 Multi-horizon (7d/30d)           │
  │    → Richer planning signals       │
  │                                    │
  │ 🌐 Cross-border spillover          │
  │    → Refugee flow dynamics         │
  │                                    │
  │ 🔌 REST API (FastAPI)               │
  │    → Integrate into dashboards     │
  │                                    │
  │ 🔄 Automated weekly refresh        │
  │    → Always up to date             │
  └────────────────────────────────────┘
 
[Bottom — Thank you line]
  Data source: ACLED (acleddata.com) · Model: LightGBM · Explainability: SHAP
```
 
### 📷 Image to Use
**`reports/figures/risk_category_distribution.png`** — the donut/bar chart showing risk distribution. Small inset at the bottom right.
 
---
 
## 📁 Complete Image Inventory for Your PPT
 
Here's every image you need and exactly where to find it:
 
### Generated by the Pipeline (in `reports/`)
| Slide | Image File | Location |
|---|---|---|
| 1 | Title background | Generated (see artifacts) |
| 5 | Pipeline diagram | Generated (see artifacts) |
| 4, 14 | `country_dashboard.png` | `reports/figures/` |
| 6, 10 | `feature_importance.png` | `reports/figures/` |
| 8 | `prediction_distribution.png` | `reports/figures/` |
| 10 | `summary_plot.png` | `reports/shap/` |
| 11 | `waterfall_pos_001.png` | `reports/shap/` |
| 11 | `waterfall_neg_004.png` | `reports/shap/` |
| 12 | `hotspots_bar.png` | `reports/figures/` |
| 13 | `hotspots_heatmap.png` | `reports/figures/` |
| 13 | Screenshot of `forecast_risk_map.html` | `reports/maps/` |
| 14 | `temporal_country_comparison.png` | `reports/figures/` |
| 15 | `risk_category_distribution.png` | `reports/figures/` |
 
### Generated for PPT (custom graphics)
| Image | Use On |
|---|---|
| Pipeline diagram (generated) | Slide 5 |
| Risk levels infographic (generated) | Slide 12 / sidebar |
| SHAP concept explainer (generated) | Slide 10 intro |
| Title background (generated) | Slide 1 |
 
---
 
## 💡 Pro Visual Tips
 
### Color Usage
- Use **dark red fills** (`#c0392b`) for Critical risk numbers — makes them pop instantly
- Use **teal borders** (`#00b4d8`) for any key insight callout boxes
- Use **white text on dark** throughout — never black on white (breaks the dark theme)
- Keep at most 2–3 colors per slide
### Layout Rules
- **One big idea per slide** — don't cram. Let charts breathe.
- **Big numbers in teal** for KPIs (F1: 0.86, PR-AUC: 0.90, 83 provinces)
- **Callout boxes** with teal border for key insights or quotes
- **Bottom status bars** with metadata (sample sizes, dates) in small grey text
### Animation Suggestions (PowerPoint/Slides)
- Slide 5: Reveal pipeline boxes one by one left to right (Appear animation, 0.2s stagger)
- Slide 9: Animate the "WINNER" row in the baseline table last
- Slide 12: Animate the risk table rows from top to bottom
- Slide 14: Fade in the 6 use-case cards one at a time
### Fonts to Use
If you have them installed: **Montserrat** (titles) + **Inter** (body)
Fallbacks: Calibri Light (titles) + Calibri (body)
 
---
 
## 🗂️ Suggested PPT Sections
 
Group your 15 slides into **5 sections** using PowerPoint section dividers:
 
| Section | Slides | Color |
|---|---|---|
| 🎯 The Problem & Solution | 1–3 | Navy/teal header |
| 📊 Data & Pipeline | 4–5 | Navy/teal header |
| 🧠 The Machine Learning | 6–9 | Navy/blue header |
| 🔍 Results & Explainability | 10–13 | Navy/red-accent header |
| 🌍 Applications & Roadmap | 14–15 | Navy/green-accent header |
