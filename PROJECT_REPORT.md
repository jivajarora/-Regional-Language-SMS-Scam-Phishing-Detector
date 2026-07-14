# Project Report: Regional-Language SMS Phishing Detector

This report serves as a professional portfolio overview of the Regional-Language SMS Phishing/Scam Detector, a consumer-protection tool targeting mobile fraud in India.

---

## 1. Problem Statement & Context

Mobile phishing and financial fraud SMS/WhatsApp alerts (KYC suspensions, fake lottery prizes, part-time job offers, and fake refund claims) target hundreds of millions of mobile users in India daily. However, existing anti-spam tools and default device filters are almost entirely English-only. 

This leaves a significant security gap: **scammers write code-mixed messages in Hindi, Hinglish (Hindi written in Roman script), and English**. Because Hinglish lacks standardized spelling and mixes English and Hindi vocabularies, standard English text classifiers fail to detect these threats, leaving non-English speaking demographics vulnerable to financial cybercrime.

This project closes this gap by building a pipeline that detects phishing messages in **Hindi, Hinglish, and English** and highlights exactly *which* words or indicators triggered the warning.

---

## 2. Technical Approach

The project was developed in three modular phases:

```
[Seed messages.json] ──► [Augmentation Script] ──► [dataset.csv]
                                                          │
   ┌──────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┐
   ▼ (Baseline Path)                                                                                             ▼ (Transformer Path)
[preprocess.py (NLP Normalization + Feature Extraction)]                                                   [Tokenizer Tokenization]
   │                                                                                                             │
[TfidfVectorizer + StandardScaler]                                                                               │
   │                                                                                                             ▼
   ▼                                                                                                 [Fine-tune MuRIL Transformer]
[Train Logistic Regression]                                                                                      │
   │                                                                                                             │
   ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   ▼
[FastAPI Serving app.py] ──► [Interactive Dark-Theme Web UI (static/index.html)]
```

### A. Data Pipeline & Normalization
To address the lack of public Hinglish/Hindi scam corpora, we developed a synthetic data-augmentation pipeline. It scales manual seed templates via random placeholder filling (swapping banks, UPI apps, amounts) and contextual synonym replacement. 
- **Hindi Normalization**: Handles Unicode NFC cleaning and halant/nukta standardization.
- **Hinglish Normalization**: Standardizes phonetic spelling variations (e.g. mapping *apka* to *aapka*, *he* to *hai*) to reduce vocabulary sparsity.
- **Feature Engineering**: Extracts explicit numeric indicators separately from text, including shortened URLs, urgency word density, credentials requests (OTP/PIN/CVV), and sender type classification (shortcodes, alpha headers, phone numbers).

### B. Baseline Classifiers
We trained a scikit-learn pipeline combining word/character TF-IDF features with scaled auxiliary features. We trained both **Logistic Regression** and **Random Forest** models. We tuned the decision threshold of the Logistic Regression model to `0.20` to maximize scam recall, ensuring false negatives (missed scams) are kept to a minimum.

### C. Multilingual Transformer Upgrade
We fine-tuned `google/muril-base-cased` (Multilingual Representations for Indian Languages) for sequence classification. MuRIL is pre-trained on abundant Indian text corpora, making it highly effective at parsing regional syntax and code-mixed spelling structures.

### D. Interpretability & Servicing
We implemented explanations for both models:
- **Baseline**: Ranks words and auxiliary features by their mathematical regression coefficients.
- **Transformer**: Captures attention weights from the final layer. We average head attentions and extract the values from the classification token `[CLS]` to all other tokens, mapping subwords back to full words to yield the top triggering words.
- The backend is served via a **FastAPI** server that feeds an interactive, dark-themed HTML/JS web demo with inline highlighted word alerts.

---

## 3. Key Evaluation Results

We evaluated both trained models against a held-out verification set of real-world message structures (`holdout_messages.json`) that were completely excluded from baseline and transformer training.

### Holdout Evaluation Metrics
Below is the side-by-side performance comparison on the held-out dataset:

| Model | Model Type | Precision (Scam) | Recall (Scam) | F1-Score (Scam) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline LR** | TF-IDF + Logistic Regression | `1.00` | `1.00` | `1.00` | `1.00` |
| **Transformer** | Fine-tuned MuRIL Model | `1.00` | `1.00` | `1.00` | `1.00` |

*Note: On this small validation scaffold, both models achieved 100% precision and recall. On larger datasets, the Transformer model's deep semantic features are expected to generalise better to structural variations, while the Baseline model remains highly lightweight and computationally inexpensive.*

---

## 4. Key Error Analysis Findings

Through a simulated and actual error analysis, we identified two primary failure patterns in regional phishing classification:
1. **Phonetic Hinglish Sparsity**: Scammers constantly modify spellings (e.g. *turant* $\rightarrow$ *trnt* $\rightarrow$ *turent*). While baseline spelling normalizers capture major spellings, highly atypical phonetic spellings bypass the TF-IDF vocabulary. MuRIL's subword tokenization is far more resilient to this issue.
2. **Contextual Negation (False Positives)**: Bank warning alerts often read *\"Bank never asks for OTP or PIN. If you receive alerts for KYC update, do not click.\"* Simple classifiers flag this as a scam (False Positive) due to high-weight keywords. Contextual models (like MuRIL) are required to capture the negative constraint (*\"never asks\"*).

---

## 5. Future Roadmap & Next Steps

If this project were scaled to a production-ready application, the next development steps would be:
1. **On-Device Android Integration**: Package the prediction service into a lightweight Android system service that listens to incoming SMS broadcast receivers, filtering messages locally on-device for user privacy.
2. **Dynamic URL Redirection Inspection**: Extend the feature extractor to dynamically follow shortened URLs in a secure sandbox, checking the final destination domain against dynamic phishing blocklists.
3. **Active Learning Feedback Loop**: Implement a feedback mechanism in the web demo where users can submit false negatives/positives to a central database, creating a continuous retraining pipeline.
