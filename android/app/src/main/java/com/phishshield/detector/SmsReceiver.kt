package com.phishshield.detector

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.telephony.SmsMessage
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.phishshield.detector.db.ScamDatabase
import com.phishshield.detector.db.ScamLog
import com.phishshield.detector.model.Classifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class SmsReceiver : BroadcastReceiver() {

    private val channelId = "scam_alerts_channel"
    private val notificationId = 1001

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != "android.provider.Telephony.SMS_RECEIVED") return

        // 1. Check if SMS Protection is enabled by user in settings
        val sharedPrefs = context.getSharedPreferences("phishshield_prefs", Context.MODE_PRIVATE)
        val isProtectionEnabled = sharedPrefs.getBoolean("sms_protection_enabled", true)
        if (!isProtectionEnabled) return

        // 2. Extract SMS messages from broadcast bundle
        val bundle: Bundle = intent.extras ?: return
        val pdus = bundle.get("pdus") as? Array<*> ?: return
        val format = bundle.getString("format")

        val smsMessages = pdus.map { pdu ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                SmsMessage.createFromPdu(pdu as ByteArray, format)
            } else {
                @Suppress("DEPRECATION")
                SmsMessage.createFromPdu(pdu as ByteArray)
            }
        }

        if (smsMessages.isEmpty()) return
        val sender = smsMessages.firstOrNull()?.originatingAddress ?: "Unknown Sender"
        val messageBody = smsMessages.joinToString("") { it.messageBody ?: "" }

        // 3. Load model JSON asset and run on-device inference
        try {
            val jsonContent = context.assets.open("model_metadata.json").bufferedReader().use { it.readText() }
            val classifier = Classifier(jsonContent)
            val result = classifier.classify(messageBody, sender)

            // 4. If classified as a scam, save logs and push notification alert
            if (result.label == "scam") {
                // Save to Room DB asynchronously
                val database = ScamDatabase.getDatabase(context)
                val logEntry = ScamLog(
                    text = messageBody,
                    sender = sender,
                    timestamp = System.currentTimeMillis(),
                    confidence = result.confidence,
                    triggeringTerms = result.topTriggeringTerms.joinToString(", ")
                )

                CoroutineScope(Dispatchers.IO).launch {
                    database.scamLogDao().insertLog(logEntry)
                }

                // Show notification alert
                showScamNotification(context, sender, result.confidence)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Builds and displays a warning notification for the flagged message.
     */
    private fun showScamNotification(context: Context, sender: String, confidence: Float) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        // Create the Notification Channel for API 26+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "SMS Scam Warnings",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Alert notifications for detected financial fraud or phishing SMS."
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(channel)
        }

        // Intent to launch MainActivity when tapped
        val launchIntent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            launchIntent,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            } else {
                PendingIntent.FLAG_UPDATE_CURRENT
            }
        )

        val confidencePercentage = (confidence * 100).toInt()
        val notification = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(android.R.drawable.stat_sys_warning) // Default system warning icon
            .setContentTitle("🚨 Possible Scam SMS Detected")
            .setContentText("From $sender ($confidencePercentage% Scam Likelihood). Tap to review.")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        try {
            // Display alert (handles POST_NOTIFICATIONS runtime checks automatically on newer versions)
            notificationManager.notify(notificationId, notification)
        } catch (e: SecurityException) {
            e.printStackTrace()
        }
    }
}
