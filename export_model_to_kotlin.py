import os
import json
import yaml
import joblib

# Load config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Path definitions
MODEL_DIR = config["paths"]["model_dir"]
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")
REPORT_PATH = config["paths"]["metrics_report_path"]
OUTPUT_JSON_PATH = os.path.join("android", "app", "src", "main", "assets", "model_metadata.json")

def export_model():
    print("Starting model export to Kotlin assets...")
    
    if not os.path.exists(LR_MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {LR_MODEL_PATH}. Please train the baseline model first.")
        
    # Load pipeline
    pipeline = joblib.load(LR_MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # 1. Extract TF-IDF Vectorizer
    vectorizer = preprocessor.named_transformers_["text"]
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_.tolist()
    
    # 2. Extract StandardScaler
    scaler = preprocessor.named_transformers_["aux"]
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    
    # Get feature list in order
    # StandardScaler handles: ['has_short_url', 'has_url', 'urgency_count', 'has_credential_request', 'has_phone_number_in_body', 'is_alpha_sender', 'is_shortcode', 'is_mobile_number']
    # Let's inspect the column transformer transformer list to make sure we match them
    aux_feature_names = preprocessor.transformers[1][2]
    
    # 3. Extract Logistic Regression
    coef = classifier.coef_[0].tolist()
    intercept = float(classifier.intercept_[0])
    
    # 4. Get optimized threshold
    threshold = 0.5
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            threshold = report["logistic_regression"]["optimized_threshold"]["threshold"]
        except Exception:
            pass
            
    # 5. Extract Hinglish Spelling Map from preprocess.py
    # We can import it or read it
    from preprocess import HINGLISH_SPELLING_MAP
    
    # 6. Extract config keywords
    urgency_kws = (
        config["keywords"]["urgency"]["english"] +
        config["keywords"]["urgency"]["hindi"] +
        config["keywords"]["urgency"]["hinglish"]
    )
    
    credential_kws = (
        config["keywords"]["credentials"]["english"] +
        config["keywords"]["credentials"]["hindi"] +
        config["keywords"]["credentials"]["hinglish"]
    )
    
    short_urls = config["short_url_domains"]
    
    # Format metadata dict
    model_metadata = {
        "decision_threshold": threshold,
        "intercept": intercept,
        "coefficients": coef, # First len(vocab) elements are TF-IDF, next len(aux) elements are scaled features
        "vocabulary": {word: int(idx) for word, idx in vocab.items()},
        "idf": idf,
        "aux_features": aux_feature_names,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "spelling_map": HINGLISH_SPELLING_MAP,
        "urgency_keywords": urgency_kws,
        "credential_keywords": credential_kws,
        "short_url_domains": short_urls
    }
    
    # Write to target directory
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
        
    print(f"Successfully exported model metadata to {OUTPUT_JSON_PATH}!")
    print(f"Total vocabulary size: {len(vocab)} words/ngrams.")
    print(f"Total model features: {len(coef)}")
    print(f"Tuned decision threshold: {threshold}")

if __name__ == "__main__":
    export_model()
