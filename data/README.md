# Data

## Raw (`raw/`)

| File | Source | Format | Pull date | Rows |
|---|---|---|---|---|
| `ACLED Data_2026-08-07_event_date_from_2017-01-07_event_date_to_2026-08-07 (1).csv` | ACLED (acleddata.com) — manual Data Export Tool download | Weekly aggregated count file (`week x country x admin1 x event_type` with `events`, `fatalities`, centroids) | 2026-08-07 | 108,945 |

### Scope handling (applied by `src/data_validation.py`)
- **Study window:** 2017-01-07 → 2026-08-07 (all rows kept; `DATE_START`/`DATE_END` in `config.py` are the configurable window).
- **Country filter:** file contains India, Myanmar, Sudan, South Sudan, Ukraine, plus two maritime entries (`Indian Ocean`: 259 rows, `Mediterranean Sea`: 173 rows). Maritime/non-country rows are filtered out by `src/data_validation.py`, leaving **108,513 rows** across **111 admin-1 units** for the four conflict-focus countries used in the pipeline (India, Myanmar, Sudan, South Sudan; Ukraine present in raw but excluded from labeled training scope).
- **Duplicates:** 0 composite-key duplicate rows found in this file.
- **Quality:** no missing values, no negative counts, no coordinate violations, no whitespace anomalies.

### Adaptation note (2026-08-07)
Per user decision, the pipeline adapts to whatever data is provided. This file is the **weekly aggregated admin-1 count** format, so:
- `geo_unit` = `admin1` (province level, not district/admin-2).
- `event_id` is a composite key (`event_date|country|admin1|event_type|sub_event_type`) because `event_id_cnty` is absent.
- `events` is the provided count column (no per-event rows).
- An event-level ACLED export (with `event_id_cnty`, `event_date`, `admin2`, `actor1`, per-event coordinates) is a **drop-in swap**: the loader detects it and canonicalizes accordingly.

## Processed (`processed/`)

| File | Contents |
|---|---|
| `cleaned_events.parquet` / `.csv` | Validated, typed, normalized event rows (one row per week/country/admin1/event-type combo) |
| `district_master.parquet` / `.csv` | One row per geo unit: admin1, country, coordinates, total events, row count, first/last event dates |

## Attribution
Data: ACLED, acleddata.com. (Free for academic use with attribution; see ACLED terms of use.)
