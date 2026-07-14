# Regional-Language SMS Phishing/Scam Detector (Phase 1 & 2)

This project is a defensive consumer-protection tool designed to detect phishing and financial fraud SMS or WhatsApp messages written in **Hindi, Hinglish (Roman-script Hindi), and English**.

It provides two core pipelines for comparison:
1. **Phase 1 Baseline**: A fast TF-IDF character/word n-gram pipeline feeding a Logistic Regression classifier, optimized for recall with engineered auxiliary features (short URLs, urgency counts, OTP/PIN/CVV requests, sender type).
2. **Phase 2 Transformer**: A fine-tuned multilingual transformer model (`google/muril-base-cased` - Multilingual Representations for Indian Languages) that captures semantic syntax and code-mixed (Hindi/Hinglish) text patterns.

---

## Technical Stack
- **Language**: Python 3.11+
- **Data Handling**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch, Hugging Face Transformers (`transformers`, `accelerate`)
- **Web App**: FastAPI, uvicorn
- **Serialization**: joblib
- **Text Normalization**: Unicode-level normalization, Hinglish spelling mapping, and optional `indic-nlp-library` integration.
- **Config Management**: PyYAML

---

## Directory Structure
```
regional-language-phishing-detector/
├── config.yaml                   # Configurable keywords, file paths, and hyperparameters
├── requirements.txt              # Pinned dependency versions
├── build_dataset.py              # Data augmentation and dataset assembly script
├── preprocess.py                 # Unicode normalizer, Hinglish standardizer, and feature extractor
├── train_model.py                # Train/test split, baseline model training, evaluation, threshold tuning
├── train_transformer.py          # Fine-tunes MuRIL model using Trainer API on CPU/GPU
├── predict_v2.py                 # Multi-model prediction service (Baseline & Transformer)
├── app.py                        # FastAPI backend server
├── static/
│   └── index.html                # Single-page web UI with inline premium styles
├── seed_messages.json            # Seed messages in Hindi, Hinglish, and English (scam & legit)
├── model_comparison.json         # Comparison metrics of baseline vs transformer
└── logs/
    ├── training.log              # Baseline training logs
    └── transformer_training.log  # Transformer training logs
```

---

## Installation & Setup

Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Running the Complete Pipeline

### 1. Data Assembly and Augmentation
Build the training dataset from the seeds in `seed_messages.json`:
```bash
python build_dataset.py
```
This generates the augmented dataset (`dataset.csv`).

### 2. Train the Baseline Model
Train the TF-IDF + Logistic Regression/Random Forest models and perform threshold tuning:
```bash
python train_model.py
```

### 3. Fine-tune the Multilingual Transformer
Fine-tune the MuRIL model on CPU or GPU (it auto-detects GPU if available) and generate side-by-side performance comparisons:
```bash
python train_transformer.py
```
This outputs `model_comparison.json`, comparing the precision, recall, F1, and accuracy of both models on the exact same train/test split.

---

## Usage Guide

### Running Predictions on the CLI
Use the unified `predict_v2.py` CLI to query messages. You can select either the `baseline` or `transformer` models:

#### Using the Fine-Tuned Transformer (MuRIL) [Default]
```bash
python predict_v2.py "प्रिय ग्राहक, आपका SBI खाता ब्लॉक कर दिया गया है। तुरंत केवाईसी अपडेट करें: http://bit.ly/kyc-update-sbi" --model transformer
```

#### Using the Baseline Model (Logistic Regression)
```bash
python predict_v2.py "Congratulations! You won a cash lottery. Call 9876543210 immediately." --model baseline --sender 9876543210
```

---

## Running the Interactive Web Demo

To launch the web interface locally:

1. Start the FastAPI server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

### Web UI Features:
- **Interactive Input**: Paste any message in English, Hindi, or Hinglish.
- **Model Selector**: Switch dynamically between the fast TF-IDF Baseline or the fine-tuned MuRIL Transformer.
- **Confidence Meter**: Shows the classification likelihood.
- **Triggering Words Highlight**: Words in the original message that triggered the flag are highlighted inline (using Unicode-aware word tokenizers in JavaScript).
- **Disclaimer**: Visible footer highlighting educational/demonstration intent.

---

## Interpretability & How Explanations Work

### 1. Baseline Model Explanation
For `baseline`, we inspect the mathematical weights (coefficients) of the trained Logistic Regression model:
- The input string is preprocessed and tokenized.
- Active features (word n-grams and engineered flags like shortened URLs) are multiplied by their coefficients.
- Positive values that drive the prediction towards `scam` are sorted to surface the top triggers.

### 2. Transformer Model Explanation
For `transformer`, we extract token-level attention weights:
- The model is loaded in `eager` mode (`attn_implementation="eager"`) to expose attention tensors.
- During inference, we capture the final-layer attention weights.
- We average the attention across all heads and extract the weights from the `[CLS]` token (which aggregates sequence information for classification) to all other tokens.
- Subword tokens (WordPieces starting with `##`) are mapped back to their root words, and their attention weights are summed.
- High-attention terms are filtered to ignore punctuation, yielding the words that most drove the transformer's decision.
