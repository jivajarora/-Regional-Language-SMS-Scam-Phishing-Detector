import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import logging
import torch

from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import preprocess_message

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Path definitions
MODEL_DIR = config["paths"]["model_dir"]
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")
TRANSFORMER_MODEL_DIR = config["paths"]["transformer_model_dir"]
REPORT_PATH = config["paths"]["metrics_report_path"]
COMPARISON_PATH = config["paths"]["model_comparison_path"]
HOLDOUT_PATH = "holdout_messages.json"
HOLDOUT_RESULTS_PATH = "holdout_evaluation_results.json"

def evaluate_baseline(messages_df, threshold):
    """
    Evaluates the Baseline Logistic Regression pipeline on the holdout set.
    """
    logger.info("Evaluating Baseline LR model on holdout set...")
    pipeline = joblib.load(LR_MODEL_PATH)
    
    y_true = messages_df["label"].map({"scam": 1, "legit": 0}).values
    preds = []
    probs = []
    
    for idx, row in messages_df.iterrows():
        sender = row.get("sender", None)
        if pd.isna(sender):
            sender = None
        else:
            sender = str(sender)
            
        cleaned_text, features_dict = preprocess_message(row["text"], sender)
        
        # Prepare input df
        input_df = pd.DataFrame([features_dict])
        input_df["cleaned_text"] = [cleaned_text]
        
        prob = float(pipeline.predict_proba(input_df)[0, 1])
        pred = 1 if prob >= threshold else 0
        
        preds.append(pred)
        probs.append(prob)
        
    report = classification_report(y_true, preds, target_names=["legit", "scam"], output_dict=True)
    cm = confusion_matrix(y_true, preds).tolist()
    
    return report, cm, preds, probs

def evaluate_transformer(messages_df):
    """
    Evaluates the fine-tuned Transformer model on the holdout set.
    """
    logger.info("Evaluating Transformer model on holdout set...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR, attn_implementation="eager")
    model.eval()
    
    y_true = messages_df["label"].map({"scam": 1, "legit": 0}).values
    preds = []
    probs = []
    
    for idx, row in messages_df.iterrows():
        inputs = tokenizer(
            row["text"], 
            return_tensors="pt", 
            truncation=True, 
            max_length=config["model"]["transformer"]["max_length"]
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        prob = float(torch.softmax(logits, dim=-1)[0, 1])
        pred = 1 if prob >= 0.5 else 0
        
        preds.append(pred)
        probs.append(prob)
        
    report = classification_report(y_true, preds, target_names=["legit", "scam"], output_dict=True)
    cm = confusion_matrix(y_true, preds).tolist()
    
    return report, cm, preds, probs

def main():
    logger.info("Starting holdout dataset evaluation...")
    
    # 1. Load holdout dataset
    if not os.path.exists(HOLDOUT_PATH):
        logger.error(f"Holdout file {HOLDOUT_PATH} not found.")
        raise FileNotFoundError(f"Holdout file {HOLDOUT_PATH} not found.")
        
    with open(HOLDOUT_PATH, "r", encoding="utf-8") as f:
        holdout_data = json.load(f)
        
    messages_df = pd.DataFrame(holdout_data)
    logger.info(f"Loaded {len(messages_df)} holdout messages.")
    
    # 2. Get baseline decision threshold
    threshold = 0.5
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            threshold = report["logistic_regression"]["optimized_threshold"]["threshold"]
        except Exception:
            pass
    logger.info(f"Using baseline decision threshold: {threshold}")
    
    # 3. Run evaluations
    lr_report, lr_cm, lr_preds, lr_probs = evaluate_baseline(messages_df, threshold)
    trans_report, trans_cm, trans_preds, trans_probs = evaluate_transformer(messages_df)
    
    # 4. Load training comparison metrics for overfitting analysis
    orig_comparison = {}
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
            orig_comparison = json.load(f)
            
    # 5. Compile results and check for overfitting
    overfitting_warnings = []
    
    # Check baseline overfitting
    if "baseline" in orig_comparison:
        orig_lr_f1 = orig_comparison["baseline"]["scam_f1"]
        hold_lr_f1 = lr_report["scam"]["f1-score"]
        if orig_lr_f1 - hold_lr_f1 > 0.15:
            warning = f"WARNING: Baseline LR F1 dropped significantly on holdout data (Original: {orig_lr_f1:.2f} vs Holdout: {hold_lr_f1:.2f}). Overfitting suspected."
            overfitting_warnings.append(warning)
            logger.warning(warning)
            
    # Check transformer overfitting
    if "transformer" in orig_comparison:
        orig_trans_f1 = orig_comparison["transformer"]["scam_f1"]
        hold_trans_f1 = trans_report["scam"]["f1-score"]
        if orig_trans_f1 - hold_trans_f1 > 0.15:
            warning = f"WARNING: Transformer F1 dropped significantly on holdout data (Original: {orig_trans_f1:.2f} vs Holdout: {hold_trans_f1:.2f}). Overfitting suspected."
            overfitting_warnings.append(warning)
            logger.warning(warning)
            
    # Compile holdout evaluation json
    holdout_results = {
        "baseline": {
            "model_type": "logistic_regression (optimized)",
            "decision_threshold": threshold,
            "metrics": lr_report,
            "confusion_matrix": lr_cm,
            "predictions": [
                {
                    "text": row["text"],
                    "true_label": row["label"],
                    "pred_label": "scam" if pred == 1 else "legit",
                    "confidence": prob,
                    "category": row.get("category", "unknown"),
                    "language": row["language"]
                }
                for (_, row), pred, prob in zip(messages_df.iterrows(), lr_preds, lr_probs)
            ]
        },
        "transformer": {
            "model_type": config["model"]["transformer"]["model_name"],
            "decision_threshold": 0.5,
            "metrics": trans_report,
            "confusion_matrix": trans_cm,
            "predictions": [
                {
                    "text": row["text"],
                    "true_label": row["label"],
                    "pred_label": "scam" if pred == 1 else "legit",
                    "confidence": prob,
                    "category": row.get("category", "unknown"),
                    "language": row["language"]
                }
                for (_, row), pred, prob in zip(messages_df.iterrows(), trans_preds, trans_probs)
            ]
        },
        "overfitting_warnings": overfitting_warnings
    }
    
    # Save holdout results
    with open(HOLDOUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(holdout_results, f, indent=2)
        
    logger.info(f"Saved holdout evaluation results to {HOLDOUT_RESULTS_PATH}")
    
    # Print side-by-side comparison on holdout
    print("\n" + "="*70)
    print("             HOLDOUT DATASET EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Holdout Samples: {len(messages_df)}")
    print("\nModel Comparisons on Holdout:")
    print(f"  {'Model':<30} | {'Scam Prec':<10} | {'Scam Rec':<10} | {'Scam F1':<10} | {'Accuracy':<8}")
    print(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
    
    print(f"  {'Baseline LR (Tuned)':<30} | {lr_report['scam']['precision']:<10.2f} | {lr_report['scam']['recall']:<10.2f} | {lr_report['scam']['f1-score']:<10.2f} | {lr_report['accuracy']:<8.2f}")
    print(f"  {'Transformer (MuRIL)':<30} | {trans_report['scam']['precision']:<10.2f} | {trans_report['scam']['recall']:<10.2f} | {trans_report['scam']['f1-score']:<10.2f} | {trans_report['accuracy']:<8.2f}")
    print("="*70 + "\n")
    
    if overfitting_warnings:
        print("\n" + "!"*70)
        print("                         OVERFITTING ALERTS")
        print("!"*70)
        for w in overfitting_warnings:
            print(f" - {w}")
        print("!"*70 + "\n")

if __name__ == "__main__":
    main()
