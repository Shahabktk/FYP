import streamlit as st
import torch
import os  
import io
from pathlib import Path
from transformers import DistilBertModel, DistilBertTokenizer 

from model import SpamClassifier
from preprocess  import clean_text, engineer_features
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading model..."):
            # 1. Define the exact local path
            model_path = Path(__file__).parent.parent / "models" / "best_spam_model.pth"
            
            # 2. Verify the local file exists
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. "
                    "Please ensure 'best_spam_model.pth' is in the 'models' folder."
                )
            
            # 3. Load the local model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 4. Initialize architecture
            bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            model = SpamClassifier(bert)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model, checkpoint['scaler'], DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()
def main():
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    st.title("✉️ Hybrid Spam Classifier")
    
    # Input Section
    col1, col2 = st.columns([3, 1])
    with col1:
        email = st.text_area("Email Text", height=200)
    with col2:
        subject = st.text_input("Subject")
        date = st.text_input("Date (YYYY-MM-DD)", "2025-07-01")
        threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.01)
    
    if st.button("Classify"):
        if not email:
            st.warning("Please enter email text!")
            return
            
        model, scaler, tokenizer = load_model()
        
        # Preprocess
        cleaned_text = clean_text(f"{subject} {email}")
        features = engineer_features(cleaned_text, subject, date, scaler)
        
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                features=torch.tensor(features).unsqueeze(0).float()
            )
            prob_spam = outputs[0, 1].item()
        
        # Display Results
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Prediction", 
                value="SPAM 🚫" if prob_spam > threshold else "HAM ✅",
                
            )
       

if __name__ == "__main__":
    main()