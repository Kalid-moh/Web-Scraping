"""
Configuration file for Bank Reviews Analysis Project
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root (folder containing Scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



# Google Play Store App IDs
APP_IDS = {
# env
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    'BOA': os.getenv('BOA_APP_ID', 'com.boa.boaMobileBanking'),
    'Dashenbank': os.getenv('DASHENBANK_APP_ID', 'com.dashen.dashensuperapp')

}

# Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'BOA': 'Bank of Abyssinia ',
    'Dashenbank': 'Dashen Bank'
}

# Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 400)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',
    'country': 'et'  # Ethiopia
}

# File Paths (absolute paths)
DATA_PATHS = {
    'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'raw_reviews': os.path.join(PROJECT_ROOT, 'data', 'raw', 'reviews_raw.csv'),
    'processed_reviews': os.path.join(PROJECT_ROOT, 'data', 'processed', 'reviews_processed.csv'),
    'sentiment_results': os.path.join(PROJECT_ROOT, 'data', 'processed', 'reviews_with_sentiment.csv'),
    'final_results': os.path.join(PROJECT_ROOT, 'data', 'processed', 'reviews_final.csv')
}










