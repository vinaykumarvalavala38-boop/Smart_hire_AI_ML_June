"""SmartHire web portal (Streamlit)."""

from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SmartHire",
    layout="wide"
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


# --------------------------------------------------
# Load ML models
# --------------------------------------------------

@st.cache_resource
def load_models():
    classifier_path = MODEL_DIR / "classifier.pkl"
    tfidf_path = MODEL_DIR / "tfidf_classifier.pkl"

    model = joblib.load(classifier_path)
    tfidf = joblib.load(tfidf_path)

    return model, tfidf


# --------------------------------------------------
# Load jobs dataset
# --------------------------------------------------

@st.cache_data
def load_jobs():
    jobs_path = DATA_DIR / "jobs_clean.csv"

    return pd.read_csv(jobs_path)


# --------------------------------------------------
# Main application
# --------------------------------------------------

st.title("SmartHire — Resume Classifier & Job Matcher")

st.write(
    "Paste your resume below to classify your profile "
    "and find the most relevant jobs."
)


# Try loading models/data
try:
    model, tfidf = load_models()
    jobs = load_jobs()

except Exception as e:
    st.error("SmartHire could not load the required files.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# Resume input
# --------------------------------------------------

resume_text = st.text_area(
    "Paste your resume text here",
    height=250
)


# --------------------------------------------------
# Analyze resume
# --------------------------------------------------

if st.button("Analyze"):

    if not resume_text.strip():

        st.warning("Please paste some resume text first.")

    else:

        try:
            # Convert resume to TF-IDF vector
            resume_vector = tfidf.transform([resume_text])

            # Predict category
            predicted_category = model.predict(resume_vector)[0]

            st.subheader("Predicted Category")
            st.success(str(predicted_category))

            # Make sure the jobs dataframe contains text
            if "text" not in jobs.columns:
                st.error(
                    "The jobs dataset does not contain a 'text' column."
                )
                st.stop()

            # Calculate similarity
            job_vectors = tfidf.transform(
                jobs["text"].fillna("").astype(str)
            )

            similarities = cosine_similarity(
                resume_vector,
                job_vectors
            ).flatten()

            # Get top 10 jobs
            top_n = min(10, len(jobs))

            top_indices = similarities.argsort()[-top_n:][::-1]

            top_jobs = jobs.iloc[top_indices].copy()

            top_jobs["match_score"] = similarities[top_indices]

            # Display results
            st.subheader("Top Matching Jobs")

            columns_to_show = [
                "title",
                "company",
                "location",
                "match_score"
            ]

            available_columns = [
                column
                for column in columns_to_show
                if column in top_jobs.columns
            ]

            st.dataframe(
                top_jobs[available_columns].reset_index(drop=True),
                use_container_width=True
            )

        except Exception as e:

            st.error("An error occurred while analyzing the resume.")
            st.exception(e)