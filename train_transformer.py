import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/transformer_training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# PyTorch Dataset implementation
class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """
    Computes precision, recall, F1, and accuracy for evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Binary evaluation for 'scam' class (class 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, 
        average='binary', 
        pos_label=1
    )
    acc = np.mean(preds == labels)
    
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def main():
    logger.info("Starting Phase 2 Transformer fine-tuning script...")
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using training device: {device.upper()}")
    
    # Load dataset
    dataset_path = config["paths"]["dataset_csv_path"]
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file {dataset_path} not found. Please run build_dataset.py first.")
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
        
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset containing {len(df)} records.")
    
    # Map target label: scam -> 1, legit -> 0
    y = df["label"].map({"scam": 1, "legit": 0}).values
    
    # Set up same stratified split
    stratify_col = df["label"].astype(str) + "_" + df["language"].astype(str)
    random_seed = config["model"]["random_seed"]
    
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, 
        test_size=0.2, 
        stratify=stratify_col, 
        random_state=random_seed
    )
    
    logger.info(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    # Load tokenizer
    model_name = config["model"]["transformer"]["model_name"]
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize inputs
    max_length = config["model"]["transformer"]["max_length"]
    logger.info("Tokenizing texts...")
    
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    
    # Create datasets
    train_dataset = SMSDataset(train_encodings, y_train)
    test_dataset = SMSDataset(test_encodings, y_test)
    
    # Load sequence classification model
    logger.info(f"Loading sequence classification model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Save directory
    transformer_model_dir = config["paths"]["transformer_model_dir"]
    os.makedirs(transformer_model_dir, exist_ok=True)
    
    # Define training arguments
    t_config = config["model"]["transformer"]
    training_args = TrainingArguments(
        output_dir=transformer_model_dir,
        num_train_epochs=t_config["epochs"],
        per_device_train_batch_size=t_config["batch_size"],
        per_device_eval_batch_size=t_config["batch_size"],
        learning_rate=float(t_config["learning_rate"]),
        weight_decay=t_config["weight_decay"],
        warmup_ratio=t_config["warmup_ratio"],
        logging_dir="./logs/transformer_runs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none", # Disable wandb/tensorboard reporting for lightweight runs
        seed=random_seed
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    # Train model
    logger.info("Starting transformer training...")
    trainer.train()
    
    # Evaluate final model
    logger.info("Evaluating final fine-tuned model...")
    eval_results = trainer.evaluate()
    
    # Run predictions on test set to get full classification report and confusion matrix
    predictions = trainer.predict(test_dataset)
    y_preds = np.argmax(predictions.predictions, axis=1)
    
    # Metrics
    report = classification_report(y_test, y_preds, target_names=["legit", "scam"], output_dict=True)
    cm = confusion_matrix(y_test, y_preds).tolist()
    
    logger.info(f"Transformer Scam Recall: {report['scam']['recall']:.4f} | Precision: {report['scam']['precision']:.4f}")
    
    # Save the fine-tuned model and tokenizer
    logger.info(f"Saving best model and tokenizer to {transformer_model_dir}...")
    trainer.save_model(transformer_model_dir)
    tokenizer.save_pretrained(transformer_model_dir)
    
    # Save metrics report
    report_data = {
        "model_name": model_name,
        "classification_report": report,
        "confusion_matrix": cm,
        "eval_metrics": eval_results
    }
    
    t_report_path = config["paths"]["transformer_metrics_report_path"]
    with open(t_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Transformer metrics written to {t_report_path}")
    
    # Create comparative model_comparison.json
    baseline_report_path = config["paths"]["metrics_report_path"]
    comparison_data = {}
    
    if os.path.exists(baseline_report_path):
        try:
            with open(baseline_report_path, "r", encoding="utf-8") as f:
                b_metrics = json.load(f)
            
            # Extract baseline metrics (optimized threshold)
            opt_lr = b_metrics["logistic_regression"]["optimized_threshold"]
            comparison_data["baseline"] = {
                "model_type": "logistic_regression (optimized)",
                "decision_threshold": opt_lr["threshold"],
                "scam_precision": opt_lr["classification_report"]["scam"]["precision"],
                "scam_recall": opt_lr["classification_report"]["scam"]["recall"],
                "scam_f1": opt_lr["classification_report"]["scam"]["f1-score"],
                "accuracy": opt_lr["classification_report"]["accuracy"]
            }
        except Exception as e:
            logger.warning(f"Could not load baseline metrics for comparison: {e}")
            
    comparison_data["transformer"] = {
        "model_type": model_name,
        "decision_threshold": 0.50, # Default sequence classification argmax
        "scam_precision": report["scam"]["precision"],
        "scam_recall": report["scam"]["recall"],
        "scam_f1": report["scam"]["f1-score"],
        "accuracy": report["accuracy"]
    }
    
    comp_path = config["paths"]["model_comparison_path"]
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    logger.info(f"Comparative report written to {comp_path}")
    
    logger.info("Transformer training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
