import os
import json
import yaml

# Path definitions
HOLDOUT_RESULTS_PATH = "holdout_evaluation_results.json"
ERROR_REPORT_PATH = "error_analysis_report.md"

def generate_report():
    if not os.path.exists(HOLDOUT_RESULTS_PATH):
        raise FileNotFoundError(f"Holdout results file {HOLDOUT_RESULTS_PATH} not found. Please run evaluate_holdout.py first.")
        
    with open(HOLDOUT_RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    baseline_preds = results["baseline"]["predictions"]
    trans_preds = results["transformer"]["predictions"]
    
    errors = []
    
    # 1. Identify baseline errors
    for pred in baseline_preds:
        if pred["pred_label"] != pred["true_label"]:
            errors.append({
                "model": "baseline_lr",
                "text": pred["text"],
                "true_label": pred["true_label"],
                "pred_label": pred["pred_label"],
                "confidence": pred["confidence"],
                "category": pred["category"],
                "language": pred["language"]
            })
            
    # 2. Identify transformer errors
    for pred in trans_preds:
        if pred["pred_label"] != pred["true_label"]:
            errors.append({
                "model": "transformer_muril",
                "text": pred["text"],
                "true_label": pred["true_label"],
                "pred_label": pred["pred_label"],
                "confidence": pred["confidence"],
                "category": pred["category"],
                "language": pred["language"]
            })
            
    # Compile report text
    report = []
    report.append("# Error Analysis Report - Regional-Language SMS Phishing Detector\n")
    report.append("This report analyzes classifications and failures from our model evaluations on the held-out real-world validation set.\n")
    
    report.append("## Executive Summary\n")
    report.append(f"Total misclassified instances across all models: **{len(errors)}**\n")
    
    # Tables for error breakdowns if errors exist
    if len(errors) > 0:
        # Group by language
        lang_counts = {}
        cat_counts = {}
        for err in errors:
            lang_counts[err["language"]] = lang_counts.get(err["language"], 0) + 1
            cat_counts[err["category"]] = cat_counts.get(err["category"], 0) + 1
            
        report.append("### Error Breakdown by Language\n")
        report.append("| Language | Error Count |")
        report.append("| :--- | :--- |")
        for lang, count in lang_counts.items():
            report.append(f"| {lang.capitalize()} | {count} |")
        report.append("\n")
        
        report.append("### Error Breakdown by Category\n")
        report.append("| Category | Error Count |")
        report.append("| :--- | :--- |")
        for cat, count in cat_counts.items():
            # clean category name for label
            clean_cat = cat.replace("_", " ").title()
            report.append(f"| {clean_cat} | {count} |")
        report.append("\n")
        
        report.append("## Detailed Error Analysis\n")
        report.append("Here is the list of specific misclassified messages and the diagnoses of why the models failed:\n")
        
        for idx, err in enumerate(errors, 1):
            report.append(f"### Error #{idx}\n")
            report.append(f"- **Message**: \"{err['text']}\"")
            report.append(f"- **Model**: `{err['model']}`")
            report.append(f"- **True Label**: `{err['true_label'].upper()}` | **Predicted Label**: `{err['pred_label'].upper()}`")
            report.append(f"- **Model Confidence**: {err['confidence']*100:.1f}%")
            report.append(f"- **Language**: `{err['language']}` | **Category**: `{err['category']}`\n")
            
            # Diagnostic note generation
            note = ""
            if err["true_label"] == "legit" and err["pred_label"] == "scam":
                # False Positive
                note = ("**Diagnosis**: False Positive. The message likely contains statistical tokens that strongly signal fraud in training "
                        "(such as links, bank names, or credential-associated words like 'verify' or 'OTP') which triggered the classification, "
                        "overriding the surrounding legitimate transactional context.")
            else:
                # False Negative
                note = ("**Diagnosis**: False Negative. The message likely employs atypical phrasing or code-mixing configurations that were "
                        "under-represented in the training data, or lacks standard urgency indicators (like 'तुरंत' or 'immediately'), "
                        "allowing the scam intent to bypass the model's warning thresholds.")
            report.append(f"{note}\n")
            report.append("---\n")
    else:
        report.append("> [!NOTE]\n")
        report.append("> Both models achieved **100% accuracy** on the provided held-out scaffolding dataset, meaning there were zero misclassified examples to report. Below is a analysis of anticipated failure modes when deploying regional-language phishing classifiers to production.\n")
        
        report.append("## Anticipated Real-World Failure Modes\n")
        report.append("When scaling this phishing detector to real-world deployment, we expect classification errors to arise from three main categories:\n")
        
        report.append("### 1. Phonetic Spelling Variations in Hinglish (Code-Mixed Text)\n")
        report.append("Hinglish has no standardized orthography. Users and scammers write Hindi words using Roman characters with huge phonetic variations. For example:\n")
        report.append("- **तुरंत** (Immediately): *turant*, *trnt*, *turent*, *toorant*\n")
        report.append("- **आपका** (Your): *aapka*, *apka*, *aapka*\n")
        report.append("While the dictionary mappings in `preprocess.py` capture the most common variations, atypical spellings or new slang may bypass the TF-IDF feature vocabulary and cause **False Negatives** (missed scams).\n\n")
        
        report.append("### 2. Semantic Negation and Legitimate Threat Warnings\n")
        report.append("Security warning messages sent by banks often contain scam-like terms. For example:\n")
        report.append("> *\"SBI never asks for OTP or PIN. If you receive calls asking for KYC, do not share. Alert: +919999888877\"*\n")
        report.append("These warnings contain multiple high-weight scam features (OTP, PIN, KYC, phone number). A simple baseline model will output a **False Positive** (flagging this warning as a scam), because it cannot parse the semantic negation (*\"never asks\"*). The MuRIL transformer, which is trained on contextual subwords, is far more resilient to this issue but still requires ample negative examples to learn warnings.\n\n")
        
        report.append("### 3. URL Masking and Redirection\n")
        report.append("Scammers continually rotate and disguise domain names. While the auxiliary feature `has_short_url` detects common shorteners (like `bit.ly`), scammers frequently use subdomains on legitimate cloud hosting platforms (like `github.io` or `vercel.app`) or buy cheap random TLDs (like `.xyz`, `.club`). \n")
        report.append("- **Effect**: Over time, static URL keyword matchers decay, resulting in **False Negatives** unless the feature extractor dynamically queries or inspects the URL redirects in real-time.\n\n")
        
        report.append("### 4. Zero-Shot Out-of-Vocabulary (OOV) Scams\n")
        report.append("If scammers invent a completely new scam hook that does not rely on KYC, lottery, refund, or jobs (e.g. fake electricity bill disconnection warnings which are currently common in India), the models may fail to flag them because the vocabulary words (like *\"electricity\"*, *\"disconnection\"*, *\"bijli\"*) were never seen during baseline training.\n")
        
    # Write report file
    with open(ERROR_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"Error analysis report successfully written to {ERROR_REPORT_PATH}")

if __name__ == "__main__":
    generate_report()
