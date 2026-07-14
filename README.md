# Regional-Language SMS Phishing/Scam Detector (Phase 1)

This project is a defensive consumer-protection tool designed to detect phishing and financial fraud SMS or WhatsApp messages written in **Hindi, Hinglish (Roman-script Hindi), and English**. 

It uses a machine learning pipeline combining **TF-IDF character/word n-grams** and **engineered auxiliary features** (like shortened URL presence, urgency keyword counts, OTP/PIN/CVV requests, and sender type analysis) to train Logistic Regression and Random Forest classifiers.

---

## Technical Stack
- **Language**: Python 3.11+
- **Data Handling**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Serialization**: joblib
- **Text Normalization**: Unicode-level normalization, Hinglish spelling mapping, and optional `indic-nlp-library` integration.
- **Config Management**: PyYAML

---

## Directory Structure
```
regional-language-phishing-detector/
├── config.yaml               # Configurable keywords, file paths, and hyperparameters
├── requirements.txt          # Pinned dependency versions
├── build_dataset.py          # Data augmentation and dataset assembly script
├── preprocess.py             # Unicode normalizer, Hinglish standardizer, and feature extractor
├── train_model.py            # Train/test split, model training, evaluation, threshold tuning
├── predict.py                # Prediction service and CLI interface with local explanations
├── seed_messages.json        # Seed messages in Hindi, Hinglish, and English (scam & legit)
└── logs/
    └── training.log          # Execution and training log file
```

---

## Getting Started

### 1. Installation
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Assembly and Data Augmentation
Generate the augmented training set `dataset.csv` from your seeds:
```bash
python build_dataset.py
```

### 3. Model Training and Threshold Tuning
Train the models, tune the decision threshold for high recall, and output performance metrics:
```bash
python train_model.py
```
This script evaluates both **Logistic Regression** and **Random Forest** models, outputs metrics to `metrics_report.json`, logs data to `logs/training.log`, and saves the trained pipelines to `models/`.

### 4. Running Predictions
Run prediction and see explanations on the command line:
```bash
python predict.py "Congratulations! You won a cash lottery of Rs. 10,000. Call 9876543210 immediately to claim." --sender "9876543210"
```
Or for regional languages:
```bash
python predict.py "प्रिय ग्राहक, आपका SBI खाता ब्लॉक कर दिया गया है। तुरंत केवाईसी अपडेट करें: http://bit.ly/kyc-update-sbi" --sender "AD-SBI"
```

---

## Detailed Operations

### How to Add Seed Examples
All seed message templates are defined in `seed_messages.json`. You can add your own examples by appending objects using this schema:
```json
  {
    "text": "Dear customer, your {bank} account has been suspended. Update KYC at {link} now.",
    "label": "scam",
    "language": "english",
    "sender": "AD-KAlert"
  }
```
* **Placeholders**: You can use placeholders like `{bank}`, `{amount}`, `{phone}`, `{link}`, `{otp}`, `{account_no}`, `{date}`, or `{ref_no}`. The augmentation pipeline will automatically replace these with randomized values from a pool to generate synthetic messages.
* **Label**: Set to `"scam"` or `"legit"`.
* **Language**: Set to `"hindi"`, `"hinglish"`, or `"english"`.
* **Sender**: Specify a sender name or number to train the sender pattern classifier (e.g. shortcodes, alpha headers, or standard numbers).

### How Synthetic Augmentation Works
The `build_dataset.py` script performs two stages of programmatic augmentation to expand seed volume defensively:
1. **Placeholder Substitution**: Randomly replaces curly-bracket placeholders (e.g., `{bank}`) with realistic entities (e.g., "SBI", "Paytm", "HDFC").
2. **Contextual Synonym Swapping**: Detects key threat words or transactional terms and swaps them with script-appropriate synonyms (e.g., "blocked" $\leftrightarrow$ "suspended", "तुरंत" $\leftrightarrow$ "जल्दी", "OTP" $\leftrightarrow$ "verification code") based on a configurable dictionary.
3. **Random Perturbation**: Varies synthetic sender IDs to match standard mobile patterns, shortcode rules, or alpha-sender formats to prevent the classifier from over-fitting.

### Model Interpretability (Trigger Terms)
When classifying a message, `predict.py` analyzes the prediction using the weights of the trained **Logistic Regression** pipeline:
- Active words or n-grams are extracted using the TF-IDF vocabulary mapping.
- Auxiliary features (like URL presence or urgency score) are extracted.
- Individual feature values are multiplied by their model coefficients.
- Positive values that drive the prediction towards the `scam` classification are sorted and displayed as **Top triggering indicators** on the CLI.
