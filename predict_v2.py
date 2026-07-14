import os
import sys
import json
import argparse
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

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import preprocess_message

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Path definitions
MODEL_DIR = config["paths"]["model_dir"]
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")
TRANSFORMER_MODEL_DIR = config["paths"]["transformer_model_dir"]
REPORT_PATH = config["paths"]["metrics_report_path"]

# Human-readable labels for engineered features in baseline model
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

# Global cache for loaded models to avoid loading overhead on every request
MODELS_CACHE = {}

def load_baseline():
    if "baseline" not in MODELS_CACHE:
        if not os.path.exists(LR_MODEL_PATH):
            raise FileNotFoundError(
                f"Trained baseline model not found at {LR_MODEL_PATH}. Please run train_model.py first."
            )
        pipeline = joblib.load(LR_MODEL_PATH)
        
        # Load optimized threshold
        threshold = 0.5
        if os.path.exists(REPORT_PATH):
            try:
                with open(REPORT_PATH, "r", encoding="utf-8") as f:
                    report = json.load(f)
                threshold = report["logistic_regression"]["optimized_threshold"]["threshold"]
            except Exception:
                pass
        MODELS_CACHE["baseline"] = (pipeline, threshold)
        
    return MODELS_CACHE["baseline"]

def load_transformer():
    if "transformer" not in MODELS_CACHE:
        if not os.path.exists(TRANSFORMER_MODEL_DIR) or not os.path.exists(os.path.join(TRANSFORMER_MODEL_DIR, "config.json")):
            raise FileNotFoundError(
                f"Fine-tuned transformer not found at {TRANSFORMER_MODEL_DIR}. Please run train_transformer.py first."
            )
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR, attn_implementation="eager")
        model.eval() # Set model to evaluation mode
        MODELS_CACHE["transformer"] = (tokenizer, model)
        
    return MODELS_CACHE["transformer"]

def predict_baseline(message: str, sender: str = None) -> dict:
    """
    Predict using the Phase 1 Baseline model (TF-IDF + Logistic Regression)
    """
    pipeline, threshold = load_baseline()
    
    cleaned_text, features_dict = preprocess_message(message, sender)
    input_df = pd.DataFrame([features_dict])
    input_df["cleaned_text"] = [cleaned_text]
    
    scam_prob = float(pipeline.predict_proba(input_df)[0, 1])
    label = "scam" if scam_prob >= threshold else "legit"
    
    # Extract baseline explanation (feature weights)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    X_transformed = preprocessor.transform(input_df)
    
    if hasattr(X_transformed, "toarray"):
        feature_values = X_transformed.toarray()[0]
    else:
        feature_values = X_transformed[0]
        
    contributions = []
    for name, coef, val in zip(feature_names, coefficients, feature_values):
        if val > 0 and coef > 0:
            score = float(coef * val)
            if name.startswith("text__"):
                friendly_name = f"'{name[6:]}'"
            elif name in AUX_FEATURE_LABELS:
                friendly_name = AUX_FEATURE_LABELS[name]
            else:
                friendly_name = name
            contributions.append({"term": friendly_name, "score": score})
            
    contributions = sorted(contributions, key=lambda x: x["score"], reverse=True)
    top_terms = [item["term"] for item in contributions[:5]]
    
    return {
        "label": label,
        "confidence": scam_prob,
        "decision_threshold": threshold,
        "top_triggering_terms": top_terms,
        "model_used": "baseline_lr"
    }

def predict_transformer(message: str) -> dict:
    """
    Predict using the Phase 2 Fine-tuned Transformer model with attention weight explanations
    """
    tokenizer, model = load_transformer()
    
    # Tokenize input message with truncation
    inputs = tokenizer(
        message, 
        return_tensors="pt", 
        truncation=True, 
        max_length=config["model"]["transformer"]["max_length"]
    )
    
    # Run inference capturing final attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions
        
    # Calculate probability via softmax
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    scam_prob = probs[1] # Class 1 is 'scam'
    
    # Argmax classification (standard 0.5 threshold)
    label = "scam" if scam_prob >= 0.5 else "legit"
    
    # --- Attention-Based Explanation ---
    # attentions[-1] is the last layer attention tensor
    # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    last_layer_attn = attentions[-1][0] # Retrieve first batch item (shape: num_heads, seq_len, seq_len)
    
    # Average across all attention heads
    mean_attention = last_layer_attn.mean(dim=0) # shape: (seq_len, seq_len)
    
    # Extract attention weights from [CLS] token (index 0) to all other tokens
    cls_attention = mean_attention[0].tolist() # shape: (seq_len,)
    
    # Retrieve tokens list
    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Aggregate subwords to word-level attention
    word_attention = {}
    current_word = ""
    current_score = 0.0
    
    for token, score in zip(tokens, cls_attention):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "<mask>"]:
            continue
            
        # Clean token for WordPiece (Hugging Face BERT/MuRIL)
        # Wordpieces start with ## to indicate continuations
        if token.startswith("##"):
            clean_token = token[2:]
            current_word += clean_token
            current_score += score
        else:
            # Save preceding word
            if current_word:
                # We save the maximum score seen for the word or sum
                word_attention[current_word.lower()] = max(word_attention.get(current_word.lower(), 0.0), current_score)
            current_word = token
            current_score = score
            
    # Save trailing word
    if current_word:
        word_attention[current_word.lower()] = max(word_attention.get(current_word.lower(), 0.0), current_score)
        
    # Filter out punctuation and very short formatting tokens
    punctuation = set([".", ",", "!", "?", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", "'", '"', "/", "\\", "@", "#", "*"])
    filtered_word_attn = [
        {"term": f"'{word}'", "score": float(score)}
        for word, score in word_attention.items()
        if word not in punctuation and len(word) > 1
    ]
    
    # Sort terms by attention score in descending order
    filtered_word_attn = sorted(filtered_word_attn, key=lambda x: x["score"], reverse=True)
    top_terms = [item["term"] for item in filtered_word_attn[:5]]
    
    return {
        "label": label,
        "confidence": scam_prob,
        "decision_threshold": 0.5,
        "top_triggering_terms": top_terms,
        "model_used": config["model"]["transformer"]["model_name"]
    }

def predict(message: str, model_type: str = "transformer", sender: str = None) -> dict:
    """
    Unified predict function that accepts a message, model type ('baseline' or 'transformer'),
    and optional sender, and returns classification and triggers.
    """
    if model_type == "baseline":
        return predict_baseline(message, sender)
    elif model_type == "transformer":
        return predict_transformer(message)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Select 'baseline' or 'transformer'.")

def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an SMS is Scam or Legitimate using Baseline or Transformer."
    )
    parser.add_argument(
        "message", 
        type=str, 
        help="The raw text of the SMS message to classify."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["baseline", "transformer"], 
        default="transformer",
        help="Model to use: 'baseline' (TF-IDF + LR) or 'transformer' (MuRIL). Default: 'transformer'."
    )
    parser.add_argument(
        "--sender", 
        type=str, 
        default=None, 
        help="Optional sender ID (used by baseline model only)."
    )
    
    args = parser.parse_args()
    
    try:
        result = predict(args.message, args.model, args.sender)
        
        print("\n" + "="*50)
        print("         SMS SCAM/PHISHING DETECTOR VERDICT (V2)")
        print("="*50)
        print(f"Message:     \"{args.message}\"")
        if args.sender and args.model == "baseline":
            print(f"Sender:      {args.sender}")
        print(f"Model Used:  {result['model_used']}")
        print("-"*50)
        
        verdict = result["label"].upper()
        conf = result["confidence"] * 100
        threshold_pct = result["decision_threshold"] * 100
        
        print(f"VERDICT:     {verdict}")
        print(f"Confidence (Scam Likelihood): {conf:.1f}%")
        print(f"Decision Threshold:           {threshold_pct:.1f}%")
        print("-"*50)
        
        if result["top_triggering_terms"]:
            print("Top triggering terms/indicators:")
            for idx, term in enumerate(result["top_triggering_terms"], 1):
                print(f"  {idx}. {term}")
        else:
            print("No significant indicators detected.")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the corresponding model first.")

if __name__ == "__main__":
    main()
