# Regional-Language SMS Phishing/Scam Detector

This project is a defensive consumer-protection tool designed to detect phishing and financial fraud SMS or WhatsApp messages written in **Hindi, Hinglish (Roman-script Hindi), and English**.

It provides two core pipelines for comparison:
1. **Phase 1 Baseline**: A fast TF-IDF character/word n-gram pipeline feeding a Logistic Regression classifier, optimized for recall with engineered auxiliary features (short URLs, urgency counts, OTP/PIN/CVV requests, sender type).
2. **Phase 2 Transformer**: A fine-tuned multilingual transformer model (`google/muril-base-cased` - Multilingual Representations for Indian Languages) that captures semantic syntax and code-mixed (Hindi/Hinglish) text patterns.
3. **Phase 4 Android Client**: A client-side Kotlin application that intercepts incoming messages on-device and applies the trained baseline classifier locally without remote connections.

---

## 📄 Project Documentation & Portfolio Reports
For detailed explanations of the engineering process, dataset curation, privacy safeguards, and diagnostic error analysis, review the following portfolio reports:
* **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: Executive summary, system architecture diagram, key evaluation tables, and development roadmap.
* **[DATASET_METHODOLOGY.md](DATASET_METHODOLOGY.md)**: In-depth details on template-sourcing, synthetic data-augmentation, distributions, and candidate limitations.
* **[error_analysis_report.md](error_analysis_report.md)**: Failure analysis report, identifying structural challenges (such as spelling standardisation gaps and semantic warnings negation) and how the transformer corrects them.
* **[PRIVACY.md](PRIVACY.md)**: Verifiable data protection guarantees, explaining how local SQLite storage and zero-network access (`INTERNET` permission omission) ensure cryptographic privacy.

---

## Technical Stack
- **Language**: Python 3.11+, Kotlin 1.9
- **Data Handling**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch, Hugging Face Transformers (`transformers`, `accelerate`)
- **Android Framework**: Jetpack Compose, Room (SQLite DB), Android SDK (Min API 26)
- **Web App**: FastAPI, uvicorn
- **Serialization**: joblib
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
├── evaluate_holdout.py           # Evaluates trained models on real held-out data
├── error_analysis.py             # Identifies misclassifications and outputs markdown report
├── export_model_to_kotlin.py     # Compiles trained LR coefficients into JSON asset for Android
├── predict_v2.py                 # Multi-model prediction service (Baseline & Transformer)
├── app.py                        # FastAPI backend server
├── static/
│   └── index.html                # Single-page web UI with inline premium styles
├── seed_messages.json            # Seed messages in Hindi, Hinglish, and English (scam & legit)
├── holdout_messages.json         # Real held-out validation messages (scam & legit)
├── model_comparison.json         # Comparison metrics of baseline vs transformer
├── holdout_evaluation_results.json # Holdout predictions output and metrics
├── error_analysis_report.md      # Auto-generated failure critique report
├── screenshots/                  # Folder for demo screenshots
├── PRIVACY.md                    # Privacy and permission guarantees
├── android/                      # Kotlin/Jetpack Compose Android project
│   ├── app/
│   │   ├── src/main/assets/      # Directory where model_metadata.json is loaded
│   │   └── src/main/java/...     # Preprocessor.kt, Classifier.kt, database and receiver classes
│   └── build.gradle.kts          # Project Gradle configs
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

### 4. Run Holdout Evaluation
Run predictions against the held-out real-world test set and compare holdout performance to train-split performance:
```bash
python evaluate_holdout.py
```

### 5. Generate Error Analysis Report
Generate the portfolio-ready markdown report detailing failure modes and causes of misclassifications:
```bash
python error_analysis.py
```

### 6. Export Model for Android On-Device Inference
Export the weights, vocabulary, spelling map, and scaler boundaries of the trained baseline model to the Android assets directory:
```bash
python export_model_to_kotlin.py
```

---

## Android App Sideloading & Build (Phase 4)

Because Google restricts incoming SMS broadcast permissions (`RECEIVE_SMS`) on Google Play to default SMS handlers, this app is built for **sideloaded local demo and testing purposes** via a connected device or emulator.

### Build and Install using Gradle Wrapper:

1. **Verify Asset Export**: Ensure you have executed the export command above to create:
   `android/app/src/main/assets/model_metadata.json`
2. **Open the Project**: Open the `android/` directory in Android Studio.
3. **Compile the Debug APK**:
   * On Windows:
     ```bash
     cd android
     gradlew.bat assembleDebug
     ```
   * On Linux/macOS:
     ```bash
     cd android
     ./gradlew assembleDebug
     ```
4. **Install the APK to connected device**:
   Ensure USB Debugging is enabled on your device (or start an emulator), and run:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```
   *(Alternatively, run `./gradlew installDebug` from the `android/` directory).*

### Running & Permission Checklist:
- Launch the **PhishShield** app.
- You will be presented with the **Consent screen** detailing the privacy guarantee (that no data leaves the device).
- Click **Enable SMS Protection** and grant the requested SMS broadcast (`RECEIVE_SMS`) and log-read (`READ_SMS`) runtime permissions.
- In settings, you can toggle protection on/off or click **Disable App Receiver** to suspend background monitoring immediately.

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
