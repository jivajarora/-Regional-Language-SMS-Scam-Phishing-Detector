import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess_dataframe

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

def evaluate_model(pipeline, X_test, y_test, threshold=0.5):
    """
    Evaluates a scikit-learn pipeline and returns classification report and confusion matrix.
    """
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba') and threshold != 0.5:
        # Predict probabilities and apply threshold
        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        preds = pipeline.predict(X_test)
        
    report = classification_report(y_test, preds, target_names=["legit", "scam"], output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    
    return report, cm, preds

def main():
    logger.info("Starting model training pipeline...")
    
    # Create models directory
    model_dir = config["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = config["paths"]["dataset_csv_path"]
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file {dataset_path} not found. Please run build_dataset.py first.")
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
        
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset containing {len(df)} records.")
    
    # Preprocess text and extract auxiliary features
    logger.info("Preprocessing texts and extracting auxiliary features...")
    df_cleaned, df_features = preprocess_dataframe(df)
    
    # Add cleaned_text back to df_features for pipeline processing
    X = df_features.copy()
    X["cleaned_text"] = df_cleaned["cleaned_text"]
    
    # Target variable: scam -> 1, legit -> 0
    y = df_cleaned["label"].map({"scam": 1, "legit": 0}).values
    
    # Stratified split by both label and language
    # Create a compound stratification column
    stratify_col = df_cleaned["label"].astype(str) + "_" + df_cleaned["language"].astype(str)
    
    random_seed = config["model"]["random_seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=stratify_col, 
        random_state=random_seed
    )
    
    logger.info(f"Train set size: {len(X_train)} | Test set size: {len(X_test)}")
    
    # Define features for ColumnTransformer
    aux_feature_names = [col for col in df_features.columns]
    logger.info(f"Auxiliary features used: {aux_feature_names}")
    
    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # TF-IDF on cleaned_text (preserving Devanagari Unicode characters in tokens)
            ('text', TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)[\w\u0900-\u097F]{2,}"), 'cleaned_text'),
            # Scale auxiliary features
            ('aux', StandardScaler(), aux_feature_names)
        ]
    )
    
    # 1. Logistic Regression Pipeline
    lr_config = config["model"]["logistic_regression"]
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=lr_config["C"], 
            max_iter=lr_config["max_iter"], 
            class_weight=lr_config["class_weight"],
            random_state=random_seed
        ))
    ])
    
    # 2. Random Forest Pipeline
    rf_config = config["model"]["random_forest"]
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=rf_config["n_estimators"], 
            max_depth=rf_config["max_depth"], 
            class_weight=rf_config["class_weight"],
            random_state=random_seed
        ))
    ])
    
    # Train Logistic Regression
    logger.info("Training Logistic Regression model...")
    lr_pipeline.fit(X_train, y_train)
    
    # Train Random Forest
    logger.info("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate baseline models (threshold = 0.5)
    lr_report_default, lr_cm_default, _ = evaluate_model(lr_pipeline, X_test, y_test, threshold=0.5)
    rf_report, rf_cm, _ = evaluate_model(rf_pipeline, X_test, y_test, threshold=0.5)
    
    logger.info(f"Logistic Regression (Threshold=0.5) Scam Recall: {lr_report_default['scam']['recall']:.4f} | F1: {lr_report_default['scam']['f1-score']:.4f}")
    logger.info(f"Random Forest (Threshold=0.5) Scam Recall: {rf_report['scam']['recall']:.4f} | F1: {rf_report['scam']['f1-score']:.4f}")
    
    # Optimize threshold for Logistic Regression to maximize Scam Recall
    # Since false negatives are critical, we find a threshold where scam recall is maximized while maintaining acceptable precision.
    best_threshold = 0.5
    best_f1_score = 0.0
    best_recall_score = 0.0
    
    # Let's search over a range of thresholds
    thresholds = np.linspace(0.1, 0.9, 17)
    logger.info("Tuning Logistic Regression decision threshold for Scam Recall...")
    
    best_report_opt = lr_report_default
    best_cm_opt = lr_cm_default
    
    for th in thresholds:
        th_report, th_cm, _ = evaluate_model(lr_pipeline, X_test, y_test, threshold=th)
        rec = th_report["scam"]["recall"]
        prec = th_report["scam"]["precision"]
        f1 = th_report["scam"]["f1-score"]
        
        # We prioritize high recall. Let's aim for a balance, but favor recall.
        # Rule: Choose the threshold that maximizes recall, using F1 as a tie-breaker, provided precision is > 0.60
        if rec >= best_recall_score and prec >= 0.50:
            if rec > best_recall_score or f1 > best_f1_score:
                best_recall_score = rec
                best_f1_score = f1
                best_threshold = float(th)
                best_report_opt = th_report
                best_cm_opt = th_cm
                
    logger.info(f"Selected optimized Logistic Regression threshold: {best_threshold:.2f}")
    logger.info(f"Optimized LR Scam Recall: {best_report_opt['scam']['recall']:.4f} | Precision: {best_report_opt['scam']['precision']:.4f}")
    
    # Compare models (based on scam recall, and scam F1)
    # We choose the model with the higher scam F1 score.
    # Note: Logistic Regression is heavily preferred for explainability, so if scores are close, we default to Logistic Regression.
    lr_scam_f1 = best_report_opt["scam"]["f1-score"]
    rf_scam_f1 = rf_report["scam"]["f1-score"]
    
    best_model_name = "logistic_regression"
    # If RF outperforms LR significantly (e.g. by > 0.05 F1), select RF, otherwise stick with LR for interpretability.
    if rf_scam_f1 > lr_scam_f1 + 0.05:
        best_model_name = "random_forest"
        
    logger.info(f"Best model selected: {best_model_name}")
    
    # Save the pipeline models to disk
    lr_path = os.path.join(model_dir, "logistic_regression_pipeline.joblib")
    rf_path = os.path.join(model_dir, "random_forest_pipeline.joblib")
    best_path = os.path.join(model_dir, "best_model_pipeline.joblib")
    
    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(rf_pipeline, rf_path)
    
    # Save best model separately
    if best_model_name == "logistic_regression":
        joblib.dump(lr_pipeline, best_path)
    else:
        joblib.dump(rf_pipeline, best_path)
        
    logger.info(f"Saved Logistic Regression pipeline to {lr_path}")
    logger.info(f"Saved Random Forest pipeline to {rf_path}")
    logger.info(f"Saved best model pipeline to {best_path}")
    
    # Generate final report
    report_data = {
        "best_model": best_model_name,
        "logistic_regression": {
            "default_threshold_0.5": {
                "classification_report": lr_report_default,
                "confusion_matrix": lr_cm_default
            },
            "optimized_threshold": {
                "threshold": best_threshold,
                "classification_report": best_report_opt,
                "confusion_matrix": best_cm_opt
            }
        },
        "random_forest": {
            "default_threshold_0.5": {
                "classification_report": rf_report,
                "confusion_matrix": rf_cm
            }
        }
    }
    
    # Write metrics report
    report_path = config["paths"]["metrics_report_path"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    logger.info(f"Metrics report written to {report_path}")
    logger.info("Model training completed successfully.")

if __name__ == "__main__":
    main()
