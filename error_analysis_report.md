# Error Analysis Report - Regional-Language SMS Phishing Detector

This report analyzes classifications and failures from our model evaluations on the held-out real-world validation set.

## Executive Summary

Total misclassified instances across all models: **0**

> [!NOTE]

> Both models achieved **100% accuracy** on the provided held-out scaffolding dataset, meaning there were zero misclassified examples to report. Below is a analysis of anticipated failure modes when deploying regional-language phishing classifiers to production.

## Anticipated Real-World Failure Modes

When scaling this phishing detector to real-world deployment, we expect classification errors to arise from three main categories:

### 1. Phonetic Spelling Variations in Hinglish (Code-Mixed Text)

Hinglish has no standardized orthography. Users and scammers write Hindi words using Roman characters with huge phonetic variations. For example:

- **तुरंत** (Immediately): *turant*, *trnt*, *turent*, *toorant*

- **आपका** (Your): *aapka*, *apka*, *aapka*

While the dictionary mappings in `preprocess.py` capture the most common variations, atypical spellings or new slang may bypass the TF-IDF feature vocabulary and cause **False Negatives** (missed scams).


### 2. Semantic Negation and Legitimate Threat Warnings

Security warning messages sent by banks often contain scam-like terms. For example:

> *"SBI never asks for OTP or PIN. If you receive calls asking for KYC, do not share. Alert: +919999888877"*

These warnings contain multiple high-weight scam features (OTP, PIN, KYC, phone number). A simple baseline model will output a **False Positive** (flagging this warning as a scam), because it cannot parse the semantic negation (*"never asks"*). The MuRIL transformer, which is trained on contextual subwords, is far more resilient to this issue but still requires ample negative examples to learn warnings.


### 3. URL Masking and Redirection

Scammers continually rotate and disguise domain names. While the auxiliary feature `has_short_url` detects common shorteners (like `bit.ly`), scammers frequently use subdomains on legitimate cloud hosting platforms (like `github.io` or `vercel.app`) or buy cheap random TLDs (like `.xyz`, `.club`). 

- **Effect**: Over time, static URL keyword matchers decay, resulting in **False Negatives** unless the feature extractor dynamically queries or inspects the URL redirects in real-time.


### 4. Zero-Shot Out-of-Vocabulary (OOV) Scams

If scammers invent a completely new scam hook that does not rely on KYC, lottery, refund, or jobs (e.g. fake electricity bill disconnection warnings which are currently common in India), the models may fail to flag them because the vocabulary words (like *"electricity"*, *"disconnection"*, *"bijli"*) were never seen during baseline training.
