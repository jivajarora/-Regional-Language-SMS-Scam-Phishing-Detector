package com.phishshield.detector.model

import android.content.Context
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.sqrt

class Classifier(jsonContent: String) {

    private val decisionThreshold: Double
    private val intercept: Double
    private val coefficients: List<Double>
    private val vocabulary: Map<String, Int>
    private val idf: List<Double>
    private val auxFeatures: List<String>
    private val scalerMean: List<Double>
    private val scalerScale: List<Double>
    private val spellingMap: Map<String, String>
    private val urgencyKeywords: List<String>
    private val credentialKeywords: List<String>
    private val shortUrlDomains: List<String>

    // Reverse vocabulary mapping for explainability (index -> word)
    private val indexToWord: Map<Int, String>

    init {
        val root = JSONObject(jsonContent)
        decisionThreshold = root.getDouble("decision_threshold")
        intercept = root.getDouble("intercept")
        
        // Parse coefficients
        val coefArray = root.getJSONArray("coefficients")
        val coefList = mutableListOf<Double>()
        for (i in 0 until coefArray.length()) {
            coefList.add(coefArray.getDouble(i))
        }
        coefficients = coefList

        // Parse vocabulary
        val vocabJson = root.getJSONObject("vocabulary")
        val vocabMap = mutableMapOf<String, Int>()
        val reverseMap = mutableMapOf<Int, String>()
        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val index = vocabJson.getInt(key)
            vocabMap[key] = index
            reverseMap[index] = key
        }
        vocabulary = vocabMap
        indexToWord = reverseMap

        // Parse IDF
        val idfArray = root.getJSONArray("idf")
        val idfList = mutableListOf<Double>()
        for (i in 0 until idfArray.length()) {
            idfList.add(idfArray.getDouble(i))
        }
        idf = idfList

        // Parse auxiliary features
        val auxArray = root.getJSONArray("aux_features")
        val auxList = mutableListOf<String>()
        for (i in 0 until auxArray.length()) {
            auxList.add(auxArray.getString(i))
        }
        auxFeatures = auxList

        // Parse scaler mean
        val meanArray = root.getJSONArray("scaler_mean")
        val meanList = mutableListOf<Double>()
        for (i in 0 until meanArray.length()) {
            meanList.add(meanArray.getDouble(i))
        }
        scalerMean = meanList

        // Parse scaler scale
        val scaleArray = root.getJSONArray("scaler_scale")
        val scaleList = mutableListOf<Double>()
        for (i in 0 until scaleArray.length()) {
            scaleList.add(scaleArray.getDouble(i))
        }
        scalerScale = scaleList

        // Parse spelling map
        val spellJson = root.getJSONObject("spelling_map")
        val spellMap = mutableMapOf<String, String>()
        val spellKeys = spellJson.keys()
        while (spellKeys.hasNext()) {
            val key = spellKeys.next()
            spellMap[key] = spellJson.getString(key)
        }
        spellingMap = spellMap

        // Parse urgency keywords
        val urgencyArray = root.getJSONArray("urgency_keywords")
        val urgencyList = mutableListOf<String>()
        for (i in 0 until urgencyArray.length()) {
            urgencyList.add(urgencyArray.getString(i))
        }
        urgencyKeywords = urgencyList

        // Parse credential keywords
        val credArray = root.getJSONArray("credential_keywords")
        val credList = mutableListOf<String>()
        for (i in 0 until credArray.length()) {
            credList.add(credArray.getString(i))
        }
        credentialKeywords = credList

        // Parse short URL domains
        val shortUrlsArray = root.getJSONArray("short_url_domains")
        val shortUrlsList = mutableListOf<String>()
        for (i in 0 until shortUrlsArray.length()) {
            shortUrlsList.add(shortUrlsArray.getString(i))
        }
        shortUrlDomains = shortUrlsList
    }

    data class Result(
        val label: String,
        val confidence: Float,
        val topTriggeringTerms: List<String>
    )

    private val auxFeatureLabels = mapOf(
        "has_short_url" to "[Shortened URL Detected]",
        "has_url" to "[Link/URL Detected]",
        "urgency_count" to "[Urgency/Threat Words]",
        "has_credential_request" to "[OTP/PIN/CVV Request]",
        "has_phone_number_in_body" to "[Phone Number in Message Body]",
        "is_alpha_sender" to "[Alpha Sender ID (e.g. AD-BANK)]",
        "is_shortcode" to "[SMS Shortcode Sender]",
        "is_mobile_number" to "[Standard Mobile Number Sender]"
    )

    /**
     * Runs full preprocessing and classifier inference on-device.
     */
    fun classify(text: String, sender: String?): Result {
        // 1. Preprocess & normalise text
        val normalHindi = Preprocessor.normalizeHindi(text)
        val normalHinglish = Preprocessor.normalizeHinglishSpelling(normalHindi, spellingMap)
        val cleanedText = Preprocessor.cleanTextForTfidf(normalHinglish)

        // 2. Extract auxiliary features
        val auxFeaturesMap = Preprocessor.extractAuxiliaryFeatures(
            text, sender, shortUrlDomains, urgencyKeywords, credentialKeywords
        )

        // 3. Tokenize cleaned text to build TF-IDF vector
        // Matches regex: [a-zA-Z\u0900-\u097F0-9_]{2,}
        val tokenRegex = Regex("[a-zA-Z\\u0900-\\u097F0-9_]{2,}")
        val tokens = tokenRegex.findAll(cleanedText).map { it.value }.toList()

        // Generate unigrams and bigrams
        val ngrams = mutableListOf<String>()
        ngrams.addAll(tokens) // unigrams
        for (i in 0 until tokens.size - 1) {
            ngrams.add("${tokens[i]} ${tokens[i+1]}") // bigrams
        }

        // Count ngrams present in vocabulary
        val tfidfVector = DoubleArray(vocabulary.size)
        for (ngram in ngrams) {
            val idx = vocabulary[ngram]
            if (idx != null) {
                tfidfVector[idx] += 1.0
            }
        }

        // Multiply by IDF weights
        for (i in tfidfVector.indices) {
            if (tfidfVector[i] > 0.0) {
                tfidfVector[i] = tfidfVector[i] * idf[i]
            }
        }

        // Apply L2 normalization
        var sumSquares = 0.0
        for (valWeight in tfidfVector) {
            sumSquares += valWeight * valWeight
        }
        val l2Norm = sqrt(sumSquares)
        if (l2Norm > 0.0) {
            for (i in tfidfVector.indices) {
                tfidfVector[i] = tfidfVector[i] / l2Norm
            }
        }

        // 4. Scale auxiliary features
        val scaledAuxVector = DoubleArray(auxFeatures.size)
        for (i in auxFeatures.indices) {
            val featureName = auxFeatures[i]
            val rawValue = auxFeaturesMap[featureName] ?: 0.0f
            val mean = scalerMean[i]
            val scale = scalerScale[i]
            scaledAuxVector[i] = (rawValue - mean) / scale
        }

        // 5. Combine vectors to build joint feature vector
        val jointVector = DoubleArray(vocabulary.size + auxFeatures.size)
        System.arraycopy(tfidfVector, 0, jointVector, 0, tfidfVector.size)
        System.arraycopy(scaledAuxVector, 0, jointVector, tfidfVector.size, scaledAuxVector.size)

        // 6. Compute dot product (z score)
        var dotProduct = 0.0
        for (i in jointVector.indices) {
            dotProduct += jointVector[i] * coefficients[i]
        }
        val z = dotProduct + intercept

        // 7. Calculate confidence (probability) via sigmoid function
        val confidence = (1.0 / (1.0 + exp(-z))).toFloat()
        val label = if (confidence >= decisionThreshold) "scam" else "legit"

        // 8. Extract triggering terms (local explanations)
        val contributions = mutableListOf<Pair<String, Double>>()
        for (i in jointVector.indices) {
            val valWeight = jointVector[i]
            val coef = coefficients[i]
            if (valWeight > 0.0 && coef > 0.0) {
                val score = valWeight * coef
                val featureName = if (i < tfidfVector.size) {
                    // It's a text ngram
                    val word = indexToWord[i] ?: ""
                    "'$word'"
                } else {
                    // It's an auxiliary feature
                    val auxIndex = i - tfidfVector.size
                    val name = auxFeatures[auxIndex]
                    auxFeatureLabels[name] ?: name
                }
                contributions.add(Pair(featureName, score))
            }
        }

        // Sort contributions in descending order of score
        val topTriggeringTerms = contributions
            .sortedByDescending { it.second }
            .take(5)
            .map { it.first }

        return Result(label, confidence, topTriggeringTerms)
    }
}
