import os
import json
import argparse
import sys
import yaml
import joblib
import pandas as pd
import numpy as np

# Avoid UnicodeEncodeError on Windows stdout/stderr
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from preprocess import preprocess_message

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Path definitions
MODEL_DIR = config["paths"]["model_dir"]
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")
REPORT_PATH = config["paths"]["metrics_report_path"]

# Human-readable mapping for auxiliary features
AUX_FEATURE_LABELS = {
    "aux__has_short_url": "[Shortened URL Detected]",
    "aux__has_url": "[Link/URL Detected]",
    "aux__urgency_count": "[Urgency/Threat Words]",
    "aux__has_credential_request": "[OTP/PIN/CVV Request]",
    "aux__has_phone_number_in_body": "[Phone Number in Message Body]",
    "aux__is_alpha_sender": "[Alpha Sender ID (e.g. AD-BANK)]",
    "aux__is_shortcode": "[SMS Shortcode Sender]",
    "aux__is_mobile_number": "[Standard Mobile Number Sender]"
}

def load_prediction_assets():
    """
    Loads the trained Logistic Regression pipeline and the optimized decision threshold.
    """
    if not os.path.exists(LR_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {LR_MODEL_PATH}. Please run train_model.py first."
        )
        
    pipeline = joblib.load(LR_MODEL_PATH)
    
    # Load optimized threshold if available
    threshold = 0.5
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            threshold = report["logistic_regression"]["optimized_threshold"]["threshold"]
        except Exception:
            pass
            
    return pipeline, threshold

def predict(message: str, sender: str = None) -> dict:
    """
    Takes a raw message and optional sender, processes it, and returns the classification
    verdict with confidence score and the top features driving the 'scam' classification.
    """
    pipeline, threshold = load_prediction_assets()
    
    # Preprocess message
    cleaned_text, features_dict = preprocess_message(message, sender)
    
    # Prepare input DataFrame for pipeline
    input_df = pd.DataFrame([features_dict])
    input_df["cleaned_text"] = [cleaned_text]
    
    # Predict probability
    # Probability of scam (class 1)
    scam_prob = float(pipeline.predict_proba(input_df)[0, 1])
    
    # Apply optimized threshold
    label = "scam" if scam_prob >= threshold else "legit"
    
    # --- Interpretability / Triggering Terms Explanation ---
    # Retrieve feature names and coefficients
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    # Transform the single input row to extract feature values
    X_transformed = preprocessor.transform(input_df)
    
    # If the preprocessor output is sparse, convert to dense or use indices
    if hasattr(X_transformed, "toarray"):
        feature_values = X_transformed.toarray()[0]
    else:
        feature_values = X_transformed[0]
        
    # Calculate feature contributions: coefficient * value
    contributions = []
    for name, coef, val in zip(feature_names, coefficients, feature_values):
        if val > 0 and coef > 0:  # Only look at active features that drive "scam"
            score = float(coef * val)
            
            # Make feature name user-friendly
            if name.startswith("text__"):
                friendly_name = f"'{name[6:]}'"
            elif name in AUX_FEATURE_LABELS:
                friendly_name = AUX_FEATURE_LABELS[name]
            else:
                friendly_name = name
                
            contributions.append({
                "term": friendly_name,
                "importance": score
            })
            
    # Sort contributions by score in descending order
    contributions = sorted(contributions, key=lambda x: x["importance"], reverse=True)
    
    # Keep only the top triggering features (e.g. top 5)
    top_triggering_terms = [item["term"] for item in contributions[:5]]
    
    return {
        "label": label,
        "confidence": scam_prob,
        "decision_threshold": threshold,
        "top_triggering_terms": top_triggering_terms
    }

def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an SMS message is a Scam or Legitimate."
    )
    parser.add_argument(
        "message", 
        type=str, 
        help="The raw text of the SMS message to classify."
    )
    parser.add_argument(
        "--sender", 
        type=str, 
        default=None, 
        help="Optional sender ID (e.g., AD-SBI, VK-HDFCBK, or a phone number)."
    )
    
    args = parser.parse_args()
    
    try:
        result = predict(args.message, args.sender)
        
        # Display the result beautifully
        print("\n" + "="*50)
        print("         SMS SCAM/PHISHING DETECTOR VERDICT")
        print("="*50)
        print(f"Message:  \"{args.message}\"")
        if args.sender:
            print(f"Sender:   {args.sender}")
        print("-"*50)
        
        # Color coding in console (optional, let's do clean text)
        verdict = result["label"].upper()
        conf = result["confidence"] * 100
        threshold_pct = result["decision_threshold"] * 100
        
        print(f"VERDICT:  {verdict}")
        print(f"Confidence (Scam Likelihood): {conf:.1f}%")
        print(f"Decision Threshold:           {threshold_pct:.1f}%")
        print("-"*50)
        
        if result["top_triggering_terms"]:
            print("Top triggering indicators for this classification:")
            for idx, term in enumerate(result["top_triggering_terms"], 1):
                print(f"  {idx}. {term}")
        else:
            print("No significant scam indicators detected.")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have run build_dataset.py and train_model.py first.")

if __name__ == "__main__":
    main()
