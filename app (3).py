
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# App Title & Styling
st.set_page_config(page_title="AI Scam & Spam Detector", page_icon="🛡️")
st.title("🛡️ AI-Powered Scam & Spam Message Detector")
st.write("Enter any SMS, Email, or Chat message below to analyze potential security risks.")

# Load and prepare dataset
@st.cache_data
def load_data():
    df = pd.read_excel('SpamSmsDataset.xlsx')
    # Clean and ensure correct column orientation
    if 'v1' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
    else:
        df = df.iloc[:, :2]
        df.columns = ['label', 'text']
    
    # Ensure correct column values
    if df['text'].isin(['ham', 'spam']).mean() > 0.5:
        df['label'], df['text'] = df['text'], df['label']
        
    df['label'] = df['label'].astype(str)
    # Fix: Ensure all text entries are strings to avoid int .lower() error
    df['text'] = df['text'].astype(str)
    return df

df = load_data()

# Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = MultinomialNB()
model.fit(X, y)

# Keywords list
SUSPICIOUS_WORDS = ['winner', 'won', 'urgent', 'bank', 'password', 'otp', 'claim', 'cash', 'prize', 'click', 'link', 'suspended']

# User Input
user_message = st.text_area("Paste Message Here:", placeholder="e.g. URGENT: Your bank account is suspended. Verify password and OTP immediately.")

if st.button("Analyze Message"):
    if user_message.strip() != "":
        # Transform and predict
        text_vec = vectorizer.transform([user_message])
        pred = model.predict(text_vec)[0]
        prob = max(model.predict_proba(text_vec)[0]) * 100
        
        # Suspicious words scan
        found = [word for word in SUSPICIOUS_WORDS if re.search(r'\b' + re.escape(word) + r'\b', user_message, re.I)]
        
        st.subheader("Analysis Results:")
        
        if pred == 'spam':
            st.error(f"🚨 **Prediction: SPAM / SCAM** (Confidence: {prob:.2f}%)")
            st.warning(f"🔍 **Suspicious Keywords Found:** {', '.join(found) if found else 'None'}")
            st.write("### 🤖 AI Explanation")
            st.write("This message contains high-risk scam triggers, urgent phrasing, or requests for sensitive information.")
            st.write("### 🛡️ Safety Recommendations")
            st.write("- Do **NOT** click any links inside this message.")
            st.write("- Never share passwords, PINs, or OTPs.")
            st.write("- Verify with official customer support directly.")
        else:
            st.success(f"✅ **Prediction: HAM (Legitimate Message)** (Confidence: {prob:.2f}%)")
            st.write("### 🤖 AI Explanation")
            st.write("This message appears safe and resembles normal everyday communication.")
            st.write("### 🛡️ Safety Recommendations")
            st.write("- No immediate security threats detected.")
    else:
        st.info("Please paste a valid message first.")
