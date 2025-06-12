import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📄 Resume Screening App")

resume_text = st.text_area("Paste the resume text below:")

if st.button("Screen Resume"):
    if resume_text.strip() == "":
        st.warning("Please paste a resume first.")
    else:
        vec = vectorizer.transform([resume_text])
        result = model.predict(vec)[0]
        if result == 1:
            st.success("✅ Selected for the next round!")
        else:
            st.error("❌ Not selected. Try improving the resume.")
