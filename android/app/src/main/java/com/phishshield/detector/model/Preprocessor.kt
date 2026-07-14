package com.phishshield.detector.model

import java.text.Normalizer
import java.util.regex.Pattern
import org.json.JSONObject

object Preprocessor {

    /**
     * Standardizes Unicode representation for Hindi text and strips zero-width joiners.
     */
    fun normalizeHindi(text: String): String {
        val nfc = Normalizer.normalize(text, Normalizer.Form.NFC)
        return nfc.replace("\u200C", "").replace("\u200D", "")
    }

    /**
     * Corrects common Hinglish spelling variations.
     */
    fun normalizeHinglishSpelling(text: String, spellingMap: Map<String, String>): String {
        val words = text.split(Pattern.compile("\\s+"))
        val normalizedWords = words.map { word ->
            // Extract clean word (alphanumeric + Devanagari)
            val cleanWord = word.replace(Regex("[^a-zA-Z\\u0900-\\u097F0-9]"), "").lowercase()
            val mapped = spellingMap[cleanWord]
            if (mapped != null) {
                // Reconstruct word with mapping (loosely keep original case style if desired)
                val isUpper = word.isNotEmpty() && word.all { it.isUpperCase() }
                val isTitle = word.isNotEmpty() && word[0].isUpperCase()
                
                val resultWord = when {
                    isUpper -> mapped.uppercase()
                    isTitle -> mapped.replaceFirstChar { it.uppercase() }
                    else -> mapped
                }
                
                // Re-add leading/trailing non-alphanumeric chars
                val prefix = word.takeWhile { !it.isLetterOrDigit() && it.code !in 0x0900..0x097F }
                val suffix = word.takeLastWhile { !it.isLetterOrDigit() && it.code !in 0x0900..0x097F }
                prefix + resultWord + suffix
            } else {
                word
            }
        }
        return normalizedWords.joinToString(" ")
    }

    /**
     * Extracts basic clean text suitable for vectorizer ngrams.
     */
    fun cleanTextForTfidf(text: String): String {
        var clean = text.lowercase()
        // Replace multiple spaces
        clean = clean.replace(Regex("\\s+"), " ")
        // Keep alphanumeric, Devanagari characters, and Indian Rupee / currency symbols
        clean = clean.replace(Regex("[^a-zA-Z0-9\\s\\u0900-\\u097F₹$]"), " ")
        // Remove duplicate spaces again
        clean = clean.replace(Regex("\\s+"), " ").trim()
        return clean
    }

    /**
     * Extracts auxiliary features matching the Python implementation.
     * All values returned are Floats, matching standard scaling requirements.
     */
    fun extractAuxiliaryFeatures(
        text: String,
        sender: String?,
        shortUrls: List<String>,
        urgencyKeywords: List<String>,
        credentialKeywords: List<String>
    ): Map<String, Float> {
        val features = mutableMapOf<String, Float>()
        val textLower = text.lowercase()

        // 1. Short URL detection
        var hasShortUrl = 0.0f
        for (domain in shortUrls) {
            val pattern = Pattern.compile("\\b" + Pattern.quote(domain) + "/")
            if (pattern.matcher(textLower).find()) {
                hasShortUrl = 1.0f
                break
            }
        }
        features["has_short_url"] = hasShortUrl

        // General URL detection
        val hasUrl = if (Pattern.compile("https?://\\S+|www\\.\\S+").matcher(textLower).find()) 1.0f else 0.0f
        features["has_url"] = hasUrl

        // 2. Urgency keyword count
        var urgencyCount = 0.0f
        for (kw in urgencyKeywords) {
            val isHindi = kw.any { it.code in 0x0900..0x097F }
            val count = if (isHindi) {
                // Simple substring count for Hindi
                textLower.split(kw).size - 1
            } else {
                // Word boundary check for English/Hinglish
                val pattern = Pattern.compile("\\b" + Pattern.quote(kw.lowercase()) + "\\b")
                val matcher = pattern.matcher(textLower)
                var matches = 0
                while (matcher.find()) {
                    matches++
                }
                matches
            }
            urgencyCount += count.toFloat()
        }
        features["urgency_count"] = urgencyCount

        // 3. Request for credentials
        var credentialCount = 0
        for (kw in credentialKeywords) {
            val isHindi = kw.any { it.code in 0x0900..0x097F }
            val count = if (isHindi) {
                textLower.split(kw).size - 1
            } else {
                val pattern = Pattern.compile("\\b" + Pattern.quote(kw.lowercase()) + "\\b")
                val matcher = pattern.matcher(textLower)
                var matches = 0
                while (matcher.find()) {
                    matches++
                }
                matches
            }
            credentialCount += count
        }
        features["has_credential_request"] = if (credentialCount > 0) 1.0f else 0.0f

        // 4. Phone number in message body
        val phonePattern = Pattern.compile("\\b(?:\\+?91)?[6789]\\d{9}\\b")
        val hasPhoneInBody = if (phonePattern.matcher(textLower).find()) 1.0f else 0.0f
        features["has_phone_number_in_body"] = hasPhoneInBody

        // 5. Sender header analysis
        var isAlphaSender = 0.0f
        var isShortcode = 0.0f
        var isMobileNumber = 0.0f

        if (sender != null) {
            val cleanSender = sender.trim()
            if (Pattern.compile("^[A-Za-z]{2}-[A-Za-z]+$").matcher(cleanSender).matches()) {
                isAlphaSender = 1.0f
            } else if (cleanSender.all { it.isDigit() } && cleanSender.length <= 6) {
                isShortcode = 1.0f
            } else if (Pattern.compile("^(?:\\+?91)?[6789]\\d{9}$").matcher(cleanSender).matches()) {
                isMobileNumber = 1.0f
            }
        }

        features["is_alpha_sender"] = isAlphaSender
        features["is_shortcode"] = isShortcode
        features["is_mobile_number"] = isMobileNumber

        return features
    }
}
