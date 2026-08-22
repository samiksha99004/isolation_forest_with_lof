# 🛡️ CIC-IDS2018 Intrusion Detection System — Project README

A two-stage machine learning pipeline that cleans 10 raw CIC-IDS2018 traffic CSVs, then trains models to detect network attacks and classify their type.

---

## 📂 Dataset

- **Source:** CIC-IDS2018 (`F:\all_10_csv`) — 10 CSV files, one per capture day
- **Total raw rows:** 16,232,943 across all 10 files
- **Format:** CICFlowMeter-extracted network flow features (80 columns per file)

---

## 🧹 Part 1: Data Cleaning Pipeline

### Step 0 — 🔍 Schema Inspection (`step0_setup_inspect.py`)
Checked all 10 CSVs for matching columns before combining them.

**Finding:** 9 files had 80 columns; the `Thuesday-20-02-2018` file had 84 — it included 4 extra identifier columns (`Flow ID`, `Src IP`, `Src Port`, `Dst IP`) that the others didn't have.

---

### Step 1 — 📥 Load + Align Schema (`step1_load_align.py`)
- Loaded all 10 CSVs in memory-safe streaming chunks (large files, ~1M+ rows each)
- Stripped whitespace from column names
- Dropped the 4 mismatched columns from the Tuesday file so all files align to 80 common columns
- Removed duplicated header rows (some files had the header row repeated mid-file from concatenated capture sessions)
- Combined everything into `combined_raw.parquet`

**Result:** 16,232,943 rows written.

---

### Step 2 — 🩹 Handle Missing / Infinite Values (`step2_missing_inf.py`)
- Dropped the `Timestamp` column (a date string, not a usable numeric feature)
- Replaced `Infinity` / `-Infinity` values with `NaN` (caused by divide-by-zero in rate columns like `Flow Byts/s`)
- **Imputed** remaining `NaN` values with each column's mean (instead of dropping rows — no data was discarded)
- Saved as `combined_no_na.parquet`

**Result:** All 16,232,943 rows preserved.

---

### Step 3 — 🔢 Fix Data Types + Deduplicate (`step3_dtypes_dedup.py`)
- Downcast `float64` columns to smaller float types to save memory
- Removed exact duplicate rows
- Saved as `combined_deduped.parquet`

**Result:** 12,222,636 rows kept (4,010,307 duplicates removed).

---

### Step 4 — 🏷️ Normalize Labels + Select Features (`step4_labels_features.py`)
- Cleaned up label spelling/casing inconsistencies
- Created a new **`Label_binary`** column: `BENIGN` vs `ATTACK`
- Kept the original detailed `Label` column intact
- Selected the 30 model-relevant feature columns
- Saved as `cleaned_final.parquet`

---

### Step 5 — 🗂️ Group Labels for the Attack Classifier (`step5_label_grouping.py`)
The raw dataset has messy, inconsistent attack names (`SSH-Bruteforce`, `Infilteration` — typo in original data, `DDOS attack-HOIC`, etc). Grouped them into clean categories matching the project's Attack Classifier design:

| Group | Includes |
|---|---|
| **DDoS** | LOIC-HTTP, HOIC, LOIC-UDP variants |
| **DoS** | Hulk, GoldenEye, Slowloris, SlowHTTPTest |
| **Bot** | Bot traffic |
| **Infiltration** | Infiltration (typo-fixed) |
| **Brute Force** | SSH-Bruteforce, FTP-BruteForce |
| **Web Attack** | Brute Force-Web, Brute Force-XSS, SQL Injection |

Added a new **`Label_group`** column with these categories. Saved as `cleaned_final_grouped.parquet` — **the final cleaned dataset** used for training.

**Final label distribution:**
```
BENIGN         10,845,848
DDoS              786,516
DoS               210,158
Bot               144,535
Infiltration      140,610
Brute Force        94,102
Web Attack            867
```

---

## 🤖 Part 2: Model Training

### 🥇 Stage 1 — Anomaly Detection (`step6f_supervised_stage1.py`)
**Goal:** Classify each flow as `BENIGN` or `ATTACK`.

**Model:** XGBoost binary classifier, trained on `Label_binary`.

> ℹ️ An unsupervised approach (density/isolation-based anomaly detection) was tried first but topped out around 60–87% accuracy — attacks like DDoS/Bot form dense, repetitive traffic clusters rather than rare scattered outliers, which those methods struggle to isolate. Switching to a supervised classifier (same labeled-data approach as Stage 2) solved this.

**Result:**
```
Accuracy: 99%
ATTACK  — precision 1.00, recall 0.90, f1 0.95
BENIGN  — precision 0.99, recall 1.00, f1 0.99
```

**Saved artifacts:**
- `xgboost_stage1_binary.joblib`
- `label_encoder_stage1.joblib`

---

### 🥈 Stage 2 — Attack Type Classification (`step7_xgboost_classifier.py`)
**Goal:** For flows flagged as `ATTACK`, identify *which* attack it is.

**Model:** XGBoost multi-class classifier, trained only on attack rows using `Label_group`.

**Result:**
```
Accuracy: ~100%
Bot          — f1 1.00
Brute Force  — f1 1.00
DDoS         — f1 1.00
DoS          — f1 1.00
Infiltration — f1 1.00
Web Attack   — f1 0.96  (smallest class, only 867 samples)
```

**Saved artifacts:**
- `xgboost_attack_classifier.joblib`
- `label_encoder.joblib`
- `scaler_stage2.joblib`

---

## ✅ Part 3: Full Pipeline Test (`step8_full_dataset_test.py`)
Ran Stage 1 on the complete cleaned dataset (all attack types mixed with benign traffic) and reported:
- ✅ Accurate detections
- ❌ Misses (real attacks called normal)
- ⚠️ False alarms (real normal traffic flagged as attack)
- 📊 Overall accuracy score

---

## 🏗️ Final Architecture

```
Network Traffic
      ↓
Feature Extraction (30 features)
      ↓
┌─────────────────────┐
│  Stage 1: XGBoost    │  → BENIGN / ATTACK
│  (binary classifier) │
└─────────────────────┘
      ↓ (if ATTACK)
┌─────────────────────┐
│  Stage 2: XGBoost    │  → DDoS / DoS / Bot /
│  (multi-class)       │     Brute Force / Web Attack / Infiltration
└─────────────────────┘
      ↓
Precision / Recall / F1 Report
```

---

## 📁 File Map

| File | Purpose |
|---|---|
| `step0_setup_inspect.py` | Check schema consistency across 10 CSVs |
| `step1_load_align.py` | Load, align schema, combine files |
| `step2_missing_inf.py` | Handle NaN/Infinity via mean imputation |
| `step3_dtypes_dedup.py` | Fix dtypes, remove duplicates |
| `step4_labels_features.py` | Normalize labels, create binary label, select features |
| `step5_label_grouping.py` | Group attack labels into classifier categories |
| `step6f_supervised_stage1.py` | Train Stage 1 (anomaly detection) model |
| `step7_xgboost_classifier.py` | Train Stage 2 (attack classifier) model |
| `step8_full_dataset_test.py` | Test full pipeline on complete dataset |
