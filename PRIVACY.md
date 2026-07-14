# Privacy & Data Protection Guarantee (PRIVACY.md)

This application is built with a **strict, compromise-free privacy model**. Mobile messages contain highly sensitive personal information, including multi-factor OTPs, bank notifications, transaction values, and private conversations. Protecting this data is a core engineering requirement, not an afterthought.

---

## 1. Zero Network Access (Cryptographically Enforced)

Unlike standard commercial scam warning apps that send SMS headers or contents to remote cloud servers for analysis, this application is **fully offline**.

* **Verifiable Security**: The `AndroidManifest.xml` file **does not declare the `android.permission.INTERNET` permission**.
* **Android OS Sandbox Enforcement**: Under the Android security model, an app without the `INTERNET` permission is physically blocked by the Linux kernel from making socket connections, sending HTTP requests, or communicating with any remote servers. 
* **Proof of Privacy**: It is mathematically and programmatically impossible for this application to exfiltrate your SMS contents.

---

## 2. On-Device Local Inference

All classification happens locally in memory on the device:
* **The Classifier**: The TF-IDF vectorizer and Logistic Regression scoring engine are implemented directly in Kotlin inside the app module.
* **Asset Loading**: The model weights, vocabulary indexes, and preprocessing mappings are loaded locally from the application's compiled assets (`model_metadata.json`).
* **Zero Telemetry**: No tracking, usage analytics, or model telemetry is collected.

---

## 3. Local-Only Storage (Room Database)

Flagged messages and classification logs are stored locally:
* **Storage Location**: Logs are written to an encrypted Room (SQLite) database stored inside the application's private filesystem directory (`/data/data/com.phishshield.detector/databases/`).
* **No Cloud Syncing**: There is no backing cloud service or synchronization channel. Deleting the app instantly and permanently wipes all message logs.
* **User Control**: You can dismiss logs or provide corrections locally. This feedback is saved strictly to local DB rows for future on-device model iterations.

---

## 4. Permission Auditing

The application requests only two system permissions:
1. **`RECEIVE_SMS`**: Required to intercept the incoming SMS broadcast receiver when a new message arrives, allowing the model to analyze the message in real-time.
2. **`READ_SMS`**: Required to read SMS database entries to present flagged logs to the user inside the local review screen.

These permissions are handled using Android's modern runtime permissions framework, meaning you can grant or revoke them at any time in your device settings.
