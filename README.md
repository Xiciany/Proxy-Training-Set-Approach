# An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction

---

## Overview

This project proposes a **feature distribution alignment** method for social bot detection in **unlabeled scenarios**.  
The core idea is to dynamically select samples from a labeled benchmark database that match the target domain distribution, thereby converting the unlabeled detection problem into a supervised classification task.

The framework achieves cross-domain adaptation through:

- Feature unification
- Tolerance-based admission
- Pseudo-alignment
- Class ratio calibration

A **fallback strategy** is embedded to ensure usability under extreme data conditions.

---

## Key Features

- **Graph‑free**: uses only user attributes and behavioral features – no social network topology required.
- **Label‑free target domain**: target data needs no annotations; training sets are built purely from feature distribution alignment.
- **Dynamically extensible**: the benchmark feature library supports continuous expansion as more datasets become available.
- **Built‑in fallback strategy**: ensures usable results even under extreme data conditions.
- **Lightweight and efficient**: significantly lower memory footprint and computation time compared to graph neural network methods.

---

## Repository Structure

```
An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction/
│
├── Baseline Feature Set/
│   ├── feature_extraction_utils.py      # Utility functions for feature derivation
│   └── knowledge_graph_inference.py     # Knowledge‑graph inference for derived features
│
├── Main Experiment/
│   ├── Twibot-20（test）.py              # Experiments on Twibot‑20
│   ├── BotSim-24（test）.py              # Experiments on BotSim‑24
│   ├── Midterm-2018（test）.py           # Experiments on Midterm‑2018
│   └── Cresci-2017（test）.py            # Experiments on Cresci‑2017
│
├── Backup Strategy/
│   ├── BotSim-24.py
│   ├── Cresci-2017.py
│   ├── Midterm-2018.py
│   └── Twibot-20.py
│
├── Input/                               # Place target CSV files here for inference
├── Output/                              # Output directory for derived features
├── data/                                # Unified data storage
│   ├── Twibot-22（part）.csv             # Benchmark source (1% of Twibot‑22)
│   ├── Twibot-20.csv
│   ├── Cresci-2017.csv
│   ├── BotSim-24.csv
│   └── midterm-2018.csv
│
└── README.md
```

---

## Datasets

| Dataset          | Size     | Human : Bot   | Source        |
|------------------|----------|---------------|---------------|
| Twibot‑22 (source) | 1,000,000 | 860,057 : 139,943 | Request only  |
| Twibot‑20        | 9,573    | 4,175 : 5,286    | Request only  |
| Cresci‑2017      | 12,737   | 3,474 : 9,263    | Public        |
| Midterm‑2018     | 50,538   | 8,092 : 42,446   | Public        |
| BotSim‑24        | 2,907    | 1,907 : 1,000    | GitHub        |

All datasets are pre‑processed into a unified CSV format with `user_id` and `label` columns  
(`0` = human, `1` = bot).

> **Note:** Due to licensing restrictions, only a **1% sample** of Twibot‑22 is provided in this repository.  
> Full datasets can be requested from their respective sources.

---

## Quick Start

### Requirements

- Python 3.8+
- Core packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tqdm`

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost tqdm
```

---

### Step 1: Prepare Data

Place all CSV files in the `data/` folder. Each file must contain `user_id` and `label` columns.

---

### Step 2: (Optional) Feature Inference

If your target dataset lacks certain features, run knowledge‑graph inference:

```bash
cd "Baseline Feature Set"
python knowledge_graph_inference.py
```

This reads all CSV files from `Input/` and writes derived features to `Output/`.

---

### Step 3: Run Main Experiments

Example: evaluate on Twibot‑20:

```bash
cd "Main Experiment"
python "Twibot-20（test）.py"
```

For other datasets, replace the script name accordingly:

- `BotSim-24（test）.py`
- `Midterm-2018（test）.py`
- `Cresci-2017（test）.py`

---

## Experimental Results

Performance of our method on three public datasets  
(95% confidence intervals shown in parentheses):

| Dataset     | Accuracy | Precision | Recall | F1        |
|-------------|----------|-----------|--------|-----------|
| Twibot‑20   | 0.8009   | 0.7526    | 0.9588 | **0.8433** (0.8365–0.8499) |
| Cresci‑2017 | 0.8650   | 0.9841    | 0.7822 | **0.8713** (0.8641–0.8791) |
| BotSim‑24   | 0.7709   | 0.6002    | 1.0000 | **0.7502** (0.7316–0.7682) |

Detailed comparisons with state‑of‑the‑art methods  
(BotDCGC, BotTrans, Bot‑MGAT, etc.) can be found in Section 4.3 of the paper.

---

## Method Pipeline

```
Target Domain Data Ingestion
         ↓
Feature Unification + Knowledge Graph Inference
         ↓
Main Path Admission Check (Platform Match, Common Features Count, Source Coverage)
         ↓
   ┌──────────┴──────────┐
   ↓                     ↓
 Conditions Met?     (No) → Fallback Strategy (Section 3.3)
   ↓ (Yes)
 Main Path (Section 3.2)
   ↓
 Optimal Feature Combination Search + Unsupervised Clustering
   ↓
 Feature Importance Calculation (XGBoost on Source Domain)
   ↓
 Multi‑cluster Dynamic Tolerance Sampling
   ↓
 Intra‑cluster Distance Selection + Quota Allocation
   ↓
 Global Class Ratio Calibration (Batch Exchange to ~1:1)
   ↓
 Pseudo‑alignment (Scaling to Eliminate Cross‑domain Scale Differences)
   ↓
 XGBoost Model Training → Target Domain Prediction
```

---

## Benchmark Feature Set Description

The benchmark feature set contains **77 feature columns** + 1 label column, organized into 11 categories.

### I. User Identifier (1 column)

| Field Name   | Data Type        | Description                              |
|--------------|------------------|------------------------------------------|
| `user_id`    | string (object)  | Unique user identifier (e.g., `u15613133`) |

### II. User Basic Information (6 columns)

| Field Name                  | Data Type    | Description                                       |
|-----------------------------|--------------|---------------------------------------------------|
| `user_age_days`             | integer      | Account age in days                               |
| `username_length`           | integer      | Length of the username string                     |
| `name_length`               | integer      | Length of the display name                        |
| `followers_count`           | integer      | Number of followers                               |
| `following_count`           | integer      | Number of accounts followed                       |
| `follower_following_ratio`  | float        | Followers / (following + 1), reflects influence   |

### III. Description Features (12 columns)

| Field Name                | Data Type | Description                                         |
|---------------------------|-----------|-----------------------------------------------------|
| `desc_length_x`           | integer   | Raw character length of description                 |
| `desc_special_char_ratio` | float     | Proportion of special characters in description     |
| `desc_has_url`            | integer   | Whether description contains a URL (0/1)            |
| `desc_url_count`          | integer   | Number of URLs in description                       |
| `desc_empty`              | integer   | Whether description is empty (0/1)                  |
| `desc_length_y`           | integer   | Alternative length measure                          |
| `word_count`              | integer   | Number of words in description                      |
| `unique_word_count`       | integer   | Number of unique words                              |
| `duplicate_word_ratio`    | float     | Proportion of repeated words                        |
| `uppercase_ratio`         | float     | Proportion of uppercase letters                     |
| `english_ratio`           | float     | Proportion of English letters                       |
| `emoji_count`             | integer   | Number of emojis in description                     |

*(Remaining categories are listed in the full paper; the above are representative.)*

### Label Column

| Field Name | Data Type | Description                 |
|------------|-----------|-----------------------------|
| `label`    | integer   | 0 = Human, 1 = Bot          |

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{wang2025proxy,
  title={An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction},
  author={Wang, Jun-Jie and Tang, Ming-Hu},
  journal={（Journal name to be confirmed）},
  year={2025}
}
```

---

## License

This code is released for **academic research purposes only**.  
For commercial use, please contact the authors.

---

## Contact

For questions or suggestions, please contact the authors.
