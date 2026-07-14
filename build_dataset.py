import os
import re
import json
import random
import yaml
import pandas as pd
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

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
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config.yaml successfully.")
except Exception as e:
    logger.error(f"Error loading config.yaml: {e}")
    raise

# Define substitution pools
SUBSTITUTION_POOLS = {
    "{bank}": ["SBI", "HDFC", "ICICI", "AXIS", "PNB", "Paytm", "PhonePe", "GooglePay", "KOTAK", "BOB"],
    "{amount}": ["Rs. 5,000", "Rs. 10,000", "Rs. 25,000", "Rs. 49,999", "Rs. 1,00,000", "10,000 INR", "Rs. 1,500"],
    "{phone}": ["+919876543210", "+918888877777", "+917000011111", "+919010203040", "09876543211", "9898989898", "+919123456789"],
    "{link}": [
        "http://bit.ly/kyc-update-sbi", "https://tinyurl.com/paytm-kyc-verify", 
        "http://short.url/sbi-block", "http://t.co/sbi_kyc_alert", 
        "http://netbanking-update.xyz", "http://refund-status-upi.org",
        "https://t.ly/bank-verification", "http://wp.me/bank-kyc"
    ],
    "{otp}": ["482019", "5821", "903412", "1092", "8743", "309281"],
    "{account_no}": ["XXXXXX1234", "XX9081", "A/c ending 4567", "3098XXXXX", "XXXXXX7890"],
    "{date}": ["14-07-2026", "today", "12th July", "2026-07-15", "15/07/2026"],
    "{ref_no}": ["Ref: UPI9081230198", "Ref No: 821908", "txn ID Txn765421", "Ref: 908123"]
}

# Synonym mapping for contextual augmentation
SYNONYMS = {
    "english": {
        "suspended": ["blocked", "deactivated", "terminated", "restricted", "disabled"],
        "update": ["verify", "submit", "validate", "confirm"],
        "immediately": ["urgently", "now", "within 24 hours", "at once"],
        "won": ["received", "been awarded", "secured"],
        "claim": ["collect", "withdraw", "receive"],
        "payment": ["bill", "amount", "dues"],
        "overdue": ["pending", "delayed", "unpaid"],
        "Do not share": ["Never share", "Don't share", "Should not share"],
        "anyone": ["third parties", "anyone else", "others"]
    },
    "hindi": {
        "ब्लॉक": ["बंद", "निलंबित", "निष्क्रिय"],
        "तुरंत": ["जल्दी", "शीघ्र", "बिना देरी किए"],
        "अपडेट करें": ["सत्यापित करें", "पूरा करें", "जमा करें"],
        "जीती है": ["प्राप्त की है", "हासिल की है"],
        "इनाम": ["पुरस्कार", "राशि"],
        "साझा न करें": ["शेयर न करें", "न बताएं"],
        "खाता": ["अकाउंट", "बैंक खाता"]
    },
    "hinglish": {
        "suspend": ["block", "band", "close", "restrict"],
        "update": ["verify", "submit", "complete"],
        "turant": ["jaldi", "abhi", "urgently", "turant hi"],
        "jeeta": ["received kiya", "won kiya", "payya"],
        "claim": ["collect", "receive", "le"],
        "share na karein": ["bataayein nahi", "share mat karein", "na share karein"],
        "account": ["ac", "a/c", "bank account"]
    }
}

def paraphrase_message(template, lang, label, used_variants):
    """
    Generate a single unique variant by substituting placeholders and synonyms
    """
    text = template
    
    # 1. First, replace any placeholders present in the template
    placeholders = re.findall(r"\{[a-zA-Z0-9_]+\}", text)
    for placeholder in placeholders:
        if placeholder in SUBSTITUTION_POOLS:
            # Choose a random value
            replacement = random.choice(SUBSTITUTION_POOLS[placeholder])
            text = text.replace(placeholder, replacement)
            
    # 2. Contextual synonym substitution
    lang_synonyms = SYNONYMS.get(lang, {})
    words = text.split()
    augmented_words = []
    for word in words:
        # Clean word punctuation for comparison (preserving Devanagari)
        clean_word = re.sub(r'[^\w\u0900-\u097F]', '', word).lower()
        replaced = False
        
        # If the word matches a synonym key, replace with a probability
        for target, choices in lang_synonyms.items():
            if target == clean_word and random.random() < 0.4:
                chosen_syn = random.choice(choices)
                # Maintain uppercase if original was uppercase or capitalized
                if word.isupper():
                    chosen_syn = chosen_syn.upper()
                elif word[0].isupper():
                    chosen_syn = chosen_syn.capitalize()
                
                # Re-add punctuation if it existed
                prefix = re.match(r'^[^\w]+', word)
                suffix = re.search(r'[^\w]+$', word)
                
                word_result = ""
                if prefix:
                    word_result += prefix.group()
                word_result += chosen_syn
                if suffix:
                    word_result += suffix.group()
                    
                augmented_words.append(word_result)
                replaced = True
                break
        
        if not replaced:
            augmented_words.append(word)
            
    augmented_text = " ".join(augmented_words)
    
    # Make sure we don't return an exact duplicate
    if augmented_text in used_variants or augmented_text == template:
        # Fallback to pure placeholder replacement without synonym replacements
        text_fallback = template
        for placeholder in placeholders:
            if placeholder in SUBSTITUTION_POOLS:
                text_fallback = text_fallback.replace(placeholder, random.choice(SUBSTITUTION_POOLS[placeholder]))
        return text_fallback
        
    return augmented_text

def build_dataset():
    logger.info("Starting dataset assembly and augmentation...")
    
    # Load seed messages
    seed_path = config["paths"]["seed_messages_path"]
    if not os.path.exists(seed_path):
        logger.error(f"Seed messages file {seed_path} not found.")
        raise FileNotFoundError(f"Seed messages file {seed_path} not found.")
        
    with open(seed_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)
        
    logger.info(f"Loaded {len(seeds)} seed messages.")
    
    # Read augmentation params
    num_scam_variants = config["augmentation"]["num_variants_per_scam"]
    num_legit_variants = config["augmentation"]["num_variants_per_legit"]
    
    # Fix random seed for reproducibility
    random.seed(config["model"]["random_seed"])
    
    records = []
    
    for seed in seeds:
        text = seed["text"]
        label = seed["label"]
        lang = seed["language"]
        sender = seed.get("sender", "")
        
        # Add the original seed
        if label == "scam":
            # Original scam seeds are labeled as 'real_scam'
            source = "real_scam"
        else:
            # Original legit seeds are labeled as 'legitimate'
            source = "legitimate"
            
        # Standardize seed text by filling placeholders with default values
        # so that it represents a real message
        seed_filled = text
        placeholders = re.findall(r"\{[a-zA-Z0-9_]+\}", seed_filled)
        for placeholder in placeholders:
            if placeholder in SUBSTITUTION_POOLS:
                seed_filled = seed_filled.replace(placeholder, SUBSTITUTION_POOLS[placeholder][0])
                
        records.append({
            "text": seed_filled,
            "label": label,
            "language": lang,
            "source": source,
            "sender": sender
        })
        
        # Generate paraphrased variants
        num_variants = num_scam_variants if label == "scam" else num_legit_variants
        used_variants = set([seed_filled])
        
        # Determine source label for augmented data
        augmented_source = "synthetic_scam" if label == "scam" else "legitimate"
        
        for _ in range(num_variants):
            variant_text = paraphrase_message(text, lang, label, used_variants)
            used_variants.add(variant_text)
            
            # Generate a slightly randomized sender for the synthetic message
            if sender:
                if sender.startswith("+91") or sender.isdigit():
                    # Random phone-like sender
                    variant_sender = f"+91{random.randint(7000000000, 9999999999)}"
                elif "-" in sender:
                    # Random alpha-sender
                    prefix = sender.split("-")[0]
                    banks = ["SBI", "HDFCBK", "ICICIB", "AXISBK", "PAYTM", "PHONEPE"]
                    variant_sender = f"{prefix}-{random.choice(banks)}"
                else:
                    variant_sender = sender
            else:
                variant_sender = ""
                
            records.append({
                "text": variant_text,
                "label": label,
                "language": lang,
                "source": augmented_source,
                "sender": variant_sender
            })
            
    df = pd.DataFrame(records)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=config["model"]["random_seed"]).reset_index(drop=True)
    
    # Save dataset
    dataset_path = config["paths"]["dataset_csv_path"]
    df.to_csv(dataset_path, index=False, encoding="utf-8")
    
    logger.info(f"Dataset generated and saved to {dataset_path}.")
    logger.info(f"Total dataset size: {len(df)}")
    
    # Log class and source counts
    class_counts = df["label"].value_counts()
    logger.info(f"Class distribution:\n{class_counts.to_string()}")
    
    source_counts = df["source"].value_counts()
    logger.info(f"Source distribution:\n{source_counts.to_string()}")
    
    lang_counts = df["language"].value_counts()
    logger.info(f"Language distribution:\n{lang_counts.to_string()}")
    
    # Detail language-by-label distribution
    lang_label = pd.crosstab(df["language"], df["label"])
    logger.info(f"Language vs Label cross-tab:\n{lang_label.to_string()}")

if __name__ == "__main__":
    build_dataset()
