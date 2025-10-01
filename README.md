# 📧 Spam Email Detection using BERT + Feature Engineering  

## 📌 Overview  
This project is my **Final Year Project (FYP)**, focused on building a **hybrid spam email detection model** that combines **BERT embeddings** with **feature engineering**. The main goal is to improve **generalization** and **robustness** against evolving spam techniques by leveraging both deep learning and engineered features.  

## 🎯 Objectives  
- Detect spam emails efficiently with **high accuracy and low inference time**.  
- Improve **generalization** by combining **semantic understanding (BERT)** with **statistical/linguistic features**.  
- Research and implement techniques from **recent spam detection studies (last 4 years)**.  

## 🔑 Key Features  
- **Hybrid Approach** → DistilBERT embeddings (768 dims) + engineered features.  
- **Engineered Features** →  
  - Text length  
  - Subject length  
  - Number of exclamations  
  - Weekend flag  
  - Subject all caps  
  - Readability score  
  - Unique word ratio  
- **Custom Classifier** → Fully Connected Neural Network (MLP) that processes combined embeddings + features.  
- **Optimized for Speed** → Ensures fast inference for real-world applications.  
- **Deployment Ready** → Interactive web app built with **Streamlit** for real-time spam detection.  


## 📊 Dataset  
- **Size**: ~33,665 entries with 13 columns.  
- **Sources**: Public datasets + engineered features.  
- **Balanced** dataset for spam/ham classification.  

## 🛠 Tools & Libraries  
- **Data Collection & Preprocessing**: Pandas, NumPy, NLTK, SpaCy, Scikit-learn  
- **Feature Engineering**: Custom Python scripts  
- **Deep Learning**: PyTorch, Hugging Face Transformers (DistilBERT)  
- **Evaluation**: Scikit-learn, Matplotlib, Seaborn  
- **Deployment (Optional)**: Flask, FastAPI, Docker  

## 🚀 Methodology  
1. Data collection & preprocessing  
2. Feature engineering (linguistic + statistical features)  
3. Text embedding using **DistilBERT**  
4. Combine embeddings + features → Hybrid input  
5. Train **SpamClassifier (MLP)**  
6. Evaluate model (Accuracy, F1-Score, ROC-AUC)  
7. Deploy for inference  

## 📈 Expected Results  
- High accuracy and recall in detecting spam.  
- Better generalization compared to models using only BERT or only features.  
- Low inference time for practical applications.  

## 🔮 Future Work  
- Extend to **multilingual spam detection**.  
- Experiment with **LLMs + fine-tuning**.  
- Enhance feature engineering for adaptive spam techniques.  

---
