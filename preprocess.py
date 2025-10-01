import re
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd

def clean_text(text):
    """Identical to your training preprocessing"""
    if pd.isna(text):
        return ""
    text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
    text = re.sub(r'[^\w\s!?.@#$%&*+\-=/:[{\]}''"]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def engineer_features(text, subject, date, scaler):
    """Identical to training feature engineering"""
    features = np.zeros(5)
    combined_text = f"{subject} {text}"
    
    # 1. Text Length (capped at 10000)
    features[0] = min(len(combined_text), 10000)
    
    # 2. Exclamation Count (capped at 50)
    features[1] = min(combined_text.count('!'), 50)
    
    # 3. Weekday (handle invalid dates)
    try:
        weekday = pd.to_datetime(date).weekday()
    except:
        weekday = -1
    features[2] = weekday
    
    # 4. Special Characters Ratio
    spam_chars = len(re.findall(r'[@#$%&*+]', combined_text))
    features[3] = spam_chars / len(combined_text) if len(combined_text) > 0 else 0
    
    # 5. Average Word Length (capped at 20)
    words = combined_text.split()
    avg_len = sum(len(word) for word in words)/len(words) if words else 0
    features[4] = min(avg_len, 20)
    
    # Scale features (same as training)
    features[[0, 1, 3, 4]] = scaler.transform([features[[0, 1, 3, 4]]])[0]
    
    return features