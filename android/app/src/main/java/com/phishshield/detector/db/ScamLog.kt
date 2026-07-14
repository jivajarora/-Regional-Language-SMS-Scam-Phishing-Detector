package com.phishshield.detector.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "scam_logs")
data class ScamLog(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val text: String,
    val sender: String,
    val timestamp: Long,
    val confidence: Float,
    val userFeedback: String = "pending", // "pending", "confirmed" (confirmed scam), "dismissed" (false alarm)
    val triggeringTerms: String // Comma-separated or JSON list of triggers (e.g. "'sbi', [Urgency Words]")
)
