import os
import re
import unicodedata
import yaml
import pandas as pd
import numpy as np

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Try importing indic-nlp-library, create robust fallback if unavailable
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    # We initialize the factory; if it fails due to missing resources later,
    # we catch the exception during runtime.
    INDIC_NORMALIZER_FACTORY = IndicNormalizerFactory()
    INDIC_HI_NORMALIZER = INDIC_NORMALIZER_FACTORY.get_normalizer("hi")
    HAS_INDIC_NLP = True
except Exception:
    HAS_INDIC_NLP = False
    INDIC_HI_NORMALIZER = None

# Common Hinglish spelling normalization dictionary
HINGLISH_SPELLING_MAP = {
    "apka": "aapka",
    "aapika": "aapka",
    "he": "hai",
    "h": "hai",
    "acc": "account",
    "acount": "account",
    "ac": "account",
    "a/c": "account",
    "blck": "block",
    "blok": "block",
    "suspendd": "suspend",
    "k.y.c": "kyc",
    "trnt": "turant",
    "jld": "jaldi",
    "lotry": "lottery",
    "paise": "paisa",
    "rupay": "rupee",
    "rupe": "rupee",
    "rupees": "rupee",
    "rs": "rupee",
    "o.t.p": "otp",
    "u.p.i": "upi",
    "c.v.v": "cvv",
    "p.i.n": "pin",
    "msg": "message",
    "messg": "message",
    "plz": "please",
    "pls": "please"
}

def normalize_hindi(text: str) -> str:
    """
    Applies Unicode normalization and Indic NLP normalizer (if available) to Hindi text.
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Remove zero-width spaces/joiners
    text = text.replace("\u200c", "").replace("\u200d", "")
    
    if HAS_INDIC_NLP and INDIC_HI_NORMALIZER is not None:
        try:
            text = INDIC_HI_NORMALIZER.normalize(text)
        except Exception:
            # Fallback to basic cleaning if resources are missing
            pass
            
    return text

def normalize_hinglish_spelling(text: str) -> str:
    """
    Standardizes common Hinglish spelling variations.
    """
    words = text.split()
    normalized_words = []
    for word in words:
        clean_word = re.sub(r'[^\w\u0900-\u097F]', '', word).lower()
        if clean_word in HINGLISH_SPELLING_MAP:
            normalized_word = HINGLISH_SPELLING_MAP[clean_word]
            # Preserve capitalization style loosely if possible
            if word.isupper():
                normalized_word = normalized_word.upper()
            elif word[0].isupper():
                normalized_word = normalized_word.capitalize()
            
            # Re-attach punctuation
            prefix = re.match(r'^[^\w]+', word)
            suffix = re.search(r'[^\w]+$', word)
            
            word_result = ""
            if prefix:
                word_result += prefix.group()
            word_result += normalized_word
            if suffix:
                word_result += suffix.group()
            normalized_words.append(word_result)
        else:
            normalized_words.append(word)
            
    return " ".join(normalized_words)

def clean_text_for_tfidf(text: str) -> str:
    """
    Performs lowercasing and basic punctuation cleaning, leaving words and basic structures.
    """
    # Lowercase
    text = text.lower()
    
    # Standardize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation but keep numbers, currency symbols, and basic letters
    # We want to replace punctuation with a space so that words are not glued together
    text = re.sub(r'[^\w\s\u0900-\u097F₹$£%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_auxiliary_features(text: str, sender: str = None) -> dict:
    """
    Extracts numerical/boolean flags from raw message text and sender field.
    All returned features must be numeric (0 or 1, or counts) to feed downstream models.
    """
    features = {}
    text_lower = text.lower()
    
    # 1. Short URL detection
    short_domains = config["short_url_domains"]
    has_short_url = 0
    # Check for direct matches of short domains (e.g. bit.ly/abc or tinyurl.com/xyz)
    for domain in short_domains:
        pattern = rf"\b{re.escape(domain)}/"
        if re.search(pattern, text_lower):
            has_short_url = 1
            break
    features["has_short_url"] = has_short_url
    
    # General URL detection
    has_url = 1 if re.search(r"https?://\S+|www\.\S+", text_lower) else 0
    features["has_url"] = has_url
    
    # 2. Urgency keyword counts
    urgency_count = 0
    all_urgency_keywords = (
        config["keywords"]["urgency"]["english"] +
        config["keywords"]["urgency"]["hindi"] +
        config["keywords"]["urgency"]["hinglish"]
    )
    for kw in all_urgency_keywords:
        # Use boundary for english/hinglish, simple check for hindi
        if re.search(r'[\u0900-\u097F]', kw):
            pattern = re.escape(kw)
        else:
            pattern = rf"\b{re.escape(kw.lower())}\b"
            
        matches = re.findall(pattern, text_lower)
        urgency_count += len(matches)
    features["urgency_count"] = urgency_count
    
    # 3. Request for sensitive credentials (OTP/PIN/CVV/Password)
    credential_count = 0
    all_credential_keywords = (
        config["keywords"]["credentials"]["english"] +
        config["keywords"]["credentials"]["hindi"] +
        config["keywords"]["credentials"]["hinglish"]
    )
    for kw in all_credential_keywords:
        if re.search(r'[\u0900-\u097F]', kw):
            pattern = re.escape(kw)
        else:
            pattern = rf"\b{re.escape(kw.lower())}\b"
            
        matches = re.findall(pattern, text_lower)
        credential_count += len(matches)
    features["has_credential_request"] = 1 if credential_count > 0 else 0
    
    # 4. Phone numbers in message body
    # Standard 10-digit number, optionally with +91 or 91
    phone_pattern = r"\b(?:\+?91)?[6789]\d{9}\b"
    phone_matches = re.findall(phone_pattern, text_lower)
    features["has_phone_number_in_body"] = 1 if len(phone_matches) > 0 else 0
    
    # 5. Sender properties (if sender is provided)
    is_alpha_sender = 0
    is_shortcode = 0
    is_mobile_number = 0
    
    if sender:
        sender_clean = sender.strip()
        # Alpha sender pattern (e.g. VK-HDFCBK, AD-SBIINB)
        if re.match(r"^[A-Za-z]{2}-[A-Za-z]+$", sender_clean):
            is_alpha_sender = 1
        # Shortcode pattern (e.g. 56767, 123456)
        elif sender_clean.isdigit() and len(sender_clean) <= 6:
            is_shortcode = 1
        # Regular phone number pattern (+91... or 10 digits starting with 6-9)
        elif re.match(r"^(?:\+?91)?[6789]\d{9}$", sender_clean):
            is_mobile_number = 1
            
    features["is_alpha_sender"] = is_alpha_sender
    features["is_shortcode"] = is_shortcode
    features["is_mobile_number"] = is_mobile_number
    
    return features

def preprocess_message(text: str, sender: str = None) -> tuple[str, dict]:
    """
    Main entry point for a single text preprocessing.
    Returns:
        cleaned_text (str): Cleaned/normalized text suitable for vectorization.
        features (dict): Engineered numerical features.
    """
    # 1. Extract engineered features from raw message
    features = extract_auxiliary_features(text, sender)
    
    # 2. Normalize Hindi text
    normalized = normalize_hindi(text)
    
    # 3. Normalize Hinglish spellings
    normalized = normalize_hinglish_spelling(normalized)
    
    # 4. Clean text for TF-IDF (remove special characters, lowercase)
    cleaned = clean_text_for_tfidf(normalized)
    
    return cleaned, features

def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses a pandas DataFrame.
    Returns:
        df_cleaned (pd.DataFrame): DataFrame containing 'cleaned_text' column.
        df_features (pd.DataFrame): DataFrame containing only numerical auxiliary features.
    """
    cleaned_texts = []
    features_list = []
    
    for idx, row in df.iterrows():
        sender = row.get("sender", None)
        # Handle nan sender
        if pd.isna(sender):
            sender = None
        else:
            sender = str(sender)
            
        cleaned, features = preprocess_message(row["text"], sender)
        cleaned_texts.append(cleaned)
        features_list.append(features)
        
    df_cleaned = df.copy()
    df_cleaned["cleaned_text"] = cleaned_texts
    
    df_features = pd.DataFrame(features_list)
    
    return df_cleaned, df_features
