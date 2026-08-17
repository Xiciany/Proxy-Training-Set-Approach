An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction
This repository contains the official implementation of the paper "An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction".

Overview
This project proposes a feature distribution alignment method for social bot detection in unlabeled scenarios. The core idea is to identify platform types and compare feature distributions, then dynamically select samples from a labeled benchmark database that align with the target domain distribution, thereby converting the unlabeled detection problem into a supervised classification task.

The framework achieves cross-domain distribution adaptation through feature unification, tolerance-based admission, pseudo-alignment, and class ratio calibration, with a fallback strategy embedded to ensure usability under extreme conditions.

Key Features
Graph‑free: Uses only user attributes and behavioral features; no social network topology required

Label‑free target domain: Target data requires no annotations; training sets are built purely from feature distribution alignment

Dynamically extensible: The benchmark feature library supports continuous expansion as more datasets become available

Built‑in fallback strategy: Ensures usable results even under extreme data conditions

Lightweight and efficient: Significantly lower memory footprint and computation time compared to graph neural network methods

Repository Structure
text
An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction/
│
├── Baseline Feature Set/              # Feature benchmark modules
│   ├── feature_extraction_utils.py    # Utility functions for feature derivation
│   └── knowledge_graph_inference.py   # Knowledge‑graph inference for derived features
│
├── Main Experiment/                   # Main pipeline (Section 3.2)
│   ├── Twibot-20（test）.py            # Experiments on Twibot‑20
│   ├── BotSim-24（test）.py            # Experiments on BotSim‑24
│   ├── Midterm-2018（test）.py         # Experiments on Midterm‑2018
│   └── Cresci-2017（test）.py          # Experiments on Cresci‑2017
│
├── Backup Strategy/                   # Fallback strategy (Section 3.3)
│   ├── BotSim-24.py
│   ├── Cresci-2017.py
│   ├── Midterm-2018.py
│   └── Twibot-20.py
│
├── Input/                             # Place target CSV files here for inference
├── Output/                            # Output directory for derived features
├── data/                              # Unified data storage
│   ├── Twibot-22（part）.csv          # Benchmark source (1% of Twibot‑22)
│   ├── Twibot-20.csv
│   ├── Cresci-2017.csv
│   ├── BotSim-24.csv
│   └── midterm-2018.csv
│
└── README.md
Datasets
Dataset	Size	Human : Bot	Source
Twibot‑22 (source)	1,000,000	860,057 : 139,943	Request only
Twibot‑20	9,573	4,175 : 5,286	Request only
Cresci‑2017	12,737	3,474 : 9,263	Public
Midterm‑2018	50,538	8,092 : 42,446	Public
BotSim‑24	2,907	1,907 : 1,000	GitHub
All datasets are pre‑processed into a unified CSV format with user_id and label (0 = human, 1 = bot).

⚠️ Note: Due to licensing restrictions, only a 1% sample of Twibot‑22 is provided. Full datasets can be requested from their respective sources.

Quick Start
Requirements
Python 3.8+

Core packages: pandas, numpy, scikit‑learn, xgboost, tqdm

bash
pip install pandas numpy scikit-learn xgboost tqdm
Step 1: Prepare Data
Place all CSV files in the data/ folder. Each file must contain user_id and label columns.

Step 2: (Optional) Feature Inference
If your target dataset lacks certain features, run knowledge‑graph inference:

bash
cd "Baseline Feature Set"
python knowledge_graph_inference.py
This reads all CSV files from Input/ and writes derived features to Output/.

Step 3: Run Main Experiments
Example: evaluate on Twibot‑20:

bash
cd "Main Experiment"
python "Twibot-20（test）.py"
For other datasets, replace the script name accordingly:

BotSim-24（test）.py

Midterm-2018（test）.py

Cresci-2017（test）.py

Experimental Results
Performance of our method on three public datasets (95% confidence intervals shown in parentheses):

Dataset	Accuracy	Precision	Recall	F1
Twibot‑20	0.8009	0.7526	0.9588	0.8433 (0.8365–0.8499)
Cresci‑2017	0.8650	0.9841	0.7822	0.8713 (0.8641–0.8791)
BotSim‑24	0.7709	0.6002	1.0000	0.7502 (0.7316–0.7682)
Detailed comparisons with state‑of‑the‑art methods (BotDCGC, BotTrans, Bot‑MGAT, etc.) can be found in Section 4.3 of the paper.

Method Pipeline
text
Target Domain Data Ingestion
           ↓
Feature Unification + Knowledge Graph Inference
           ↓
Main Path Admission Check
   (Platform Match, Common Features Count, Source Coverage)
           ↓
    ┌──────────────────┐
    │  Conditions Met? │
    └──────────────────┘
         ↓Yes              ↓No
    Main Path        Fallback Strategy
    (Section 3.2)    (Section 3.3)
         ↓
Optimal Feature Combination Search + Unsupervised Clustering
         ↓
Feature Importance Calculation (XGBoost on Source Domain)
         ↓
Multi‑cluster Dynamic Tolerance Sampling
         ↓
Intra‑cluster Distance Selection + Quota Allocation
         ↓
Global Class Ratio Calibration (Batch Exchange to 1:1)
         ↓
Pseudo‑alignment (Scaling to Eliminate Cross‑domain Scale Differences)
         ↓
XGBoost Model Training → Target Domain Prediction
Benchmark Feature Set Description
The benchmark feature set contains 77 feature columns + 1 label column, organized into 11 categories.

I. User Identifier (1 column)
Field Name	Data Type	Description
user_id	string (object)	Unique user identifier, typically prefixed with 'u' (e.g., u15613133)
II. User Basic Information (6 columns)
Field Name	Data Type	Description
user_age_days	integer (int64)	Account age in days (may be negative for timestamp anomalies)
username_length	integer (int64)	Length of the username string
name_length	integer (int64)	Length of the display name
followers_count	integer (int64)	Number of followers
following_count	integer (int64)	Number of accounts followed
follower_following_ratio	float (float64)	Followers / (following + 1), reflects social influence
III. Description Features (12 columns)
Field Name	Data Type	Description
desc_length_x	integer (int64)	Raw character length of the description
desc_special_char_ratio	float (float64)	Proportion of special characters in description
desc_has_url	integer (int64)	Whether description contains a URL (0=No, 1=Yes)
desc_url_count	integer (int64)	Number of URLs in description
desc_empty	integer (int64)	Whether description is empty (0=Not empty, 1=Empty)
desc_length_y	integer (int64)	Alternative length measure (from a different processing method)
word_count	integer (int64)	Number of words in description
unique_word_count	integer (int64)	Number of unique words in description
duplicate_word_ratio	float (float64)	Proportion of repeated words
uppercase_ratio	float (float64)	Proportion of uppercase letters in description
english_ratio	float (float64)	Proportion of English letters in description
emoji_count	integer (int64)	Number of emojis in description
IV. Description Analysis (continued) (3 columns)
Field Name	Data Type	Description
special_char_count	integer (int64)	Total count of special characters in description
special_char_ratio	float (float64)	Special character ratio (may overlap with desc_special_char_ratio)
url_count	integer (int64)	Number of URLs in description (may overlap with desc_url_count)
V. Tweet Content & Behavioral Features (16 columns)
Field Name	Data Type	Description
tweet_count	integer (int64)	Total historical tweets (originals, retweets, replies)
avg_tweets_per_day	float (float64)	Average tweets per day (tweet_count / user_age_days)
avg_tweet_length	float (float64)	Average character length per tweet
avg_hashtag_count	float (float64)	Average number of hashtags per tweet
avg_symbol_count	float (float64)	Average number of symbols (e.g., $, %, &) per tweet
avg_user_mention_count	float (float64)	Average number of @mentions per tweet
avg_url_count	float (float64)	Average number of URLs per tweet
avg_media_count	float (float64)	Average number of media files (images/videos) per tweet
avg_text_special_ratio	float (float64)	Average proportion of special characters in tweet text
avg_likes	float (float64)	Average likes per tweet
avg_retweets	float (float64)	Average retweets per tweet
avg_replies	float (float64)	Average replies per tweet
avg_quotes	float (float64)	Average quote counts per tweet
VI. Tweet Type & Device Features (10 columns)
Field Name	Data Type	Description
reply_ratio	float (float64)	Proportion of reply-type tweets
referenced_tweets_ratio	float (float64)	Proportion of tweets referencing other tweets
attachments_ratio	float (float64)	Proportion of tweets with media attachments
context_annotations_ratio	float (float64)	Proportion of tweets with context annotations (e.g., topic classification)
geo_ratio	float (float64)	Proportion of tweets with geo-location tags
sensitive_ratio	float (float64)	Proportion of tweets marked as sensitive content
withheld_ratio	float (float64)	Proportion of tweets withheld due to regional policies
android_ratio	float (float64)	Proportion of tweets posted via Android
ios_ratio	float (float64)	Proportion of tweets posted via iOS
web_ratio	float (float64)	Proportion of tweets posted via web
VII. Temporal Behavior Features (7 columns)
Field Name	Data Type	Description
weekend_tweet_ratio	float (float64)	Proportion of tweets posted on weekends
min_interval_seconds	float (float64)	Minimum interval between consecutive tweets (seconds)
avg_interval_seconds	float (float64)	Average interval between consecutive tweets (seconds)
median_interval_seconds	float (float64)	Median interval between consecutive tweets (seconds)
std_interval_seconds	float (float64)	Standard deviation of tweet intervals (volatility)
cv_interval	float (float64)	Coefficient of variation (std / mean), reflects posting regularity
iqr_interval	float (float64)	Interquartile range of tweet intervals
VIII. Account Status & Verification Features (7 columns)
Field Name	Data Type	Description
is_verified	integer (int64)	Whether the account is verified (0=No, 1=Yes)
has_location	integer (int64)	Whether location is filled in profile (0=No, 1=Yes)
location_length	integer (int64)	Length of the location text
has_url	integer (int64)	Whether profile contains a URL (0=No, 1=Yes)
profile_image_is_default	integer (int64)	Whether using default avatar (0=No, 1=Yes)
has_pinned_tweet	integer (int64)	Whether a tweet is pinned (0=No, 1=Yes)
is_protected	integer (int64)	Whether the account is protected (private) (0=No, 1=Yes)
profile_url_count	integer (int64)	Number of URLs in profile
IX. Description Keyword Recognition (4 columns)
Field Name	Data Type	Description
has_career	integer (int64)	Description contains career keywords (e.g., "CEO", "Developer") (0=No, 1=Yes)
has_marketing	integer (int64)	Description contains marketing keywords (0=No, 1=Yes)
has_random_chars	integer (int64)	Description contains random/gibberish characters (0=No, 1=Yes)
has_avatar	float (float64)	Whether avatar exists (may be a probability value)
X. Category Encoding Features (11 columns)
Field Name	Data Type	Description
C	integer (int64)	Whether the user expresses being a real person (matches column C in original analysis script)
D	integer (int64)	Sentiment score or comprehensive rating (numeric, may be positive or negative)
A_01 ~ A_08	integer (int64)	One‑hot encoded description categories, 8 binary features (0/1). Corresponds to 8 description types:
Code	Description Type
01	Career & Academic Identity
02	Personal Identity & Interest Expression
03	Humor / Creativity / Personalization
04	Socio‑political & Value Advocacy
05	Content Creator / Promotion
06	Organization / Institution / Project
07	Empty or Minimalist Placeholder
08	Other Types
XI. Label Column (1 column)
Field Name	Data Type	Description
label	integer (int64)	Ground truth label: 0 = Human, 1 = Bot. Used for supervised learning training and evaluation
Citation
If you use this code or find our work useful, please cite:

bibtex
@article{wang2025proxy,
  title={An Approach for Unlabeled Social Bot Detection via Proxy Training Set Construction},
  author={Wang, Jun-Jie and Tang, Ming-Hu},
  journal={（Journal name to be confirmed）},
  year={2025}
}
License
This code is released for academic research purposes only. For commercial use, please contact the authors.

Contact
For questions or suggestions, please contact the authors.
