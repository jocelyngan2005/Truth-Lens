import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup
import io
import os
import base64
import random
import feedback_db
import time

# Set page config
st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="centered"
)

# Title and description
st.title("TruthLens.")
st.markdown("""
This application helps you determine if a news article or text might be fake news.
Choose your preferred input method below to analyze the content.
""")

# Malaysian reporting channels
REPORTING_CHANNELS = {
    "MyCheck": "https://mycheck.my/",
    "Sebenarnya.my": "https://sebenarnya.my/",
    "PDRM Cyber Crime": "https://www.rmp.gov.my/cybercrime",
    "MCMC": "https://aduan.skmm.gov.my/"
}

# Report Function
def report_button(name, url):
    st.link_button(f"Report to {name}", url, use_container_width=True)

def show_report_options():
    st.markdown("### 🚨 Report Fake News")
    st.markdown("""
    If you believe this is fake news, you can report it through these official Malaysian channels:
    """)
    
    st.markdown("#### Official Fact-Checking Platforms")
    col1, col2 = st.columns([1, 1])
    with col1:
        report_button(list(REPORTING_CHANNELS.items())[0][0], list(REPORTING_CHANNELS.items())[0][1])
    with col2:
        report_button(list(REPORTING_CHANNELS.items())[1][0], list(REPORTING_CHANNELS.items())[1][1])
    
    st.markdown("#### Law Enforcement")
    col3, col4 = st.columns(2)
    with col3:
        report_button(list(REPORTING_CHANNELS.items())[2][0], list(REPORTING_CHANNELS.items())[2][1])
    with col4:
        report_button(list(REPORTING_CHANNELS.items())[3][0], list(REPORTING_CHANNELS.items())[3][1])
    
    st.markdown("""
    ---
    **Note:** When reporting, please include:
    - The original content
    - Where you found it
    - Why you believe it's fake news
    - Any supporting evidence
    """)

# User Feedback System
def show_feedback_system():
    st.markdown("### 💬 Was this prediction helpful?")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("👍 Like", key="upvote_btn", use_container_width=True):
            st.session_state["feedback_type"] = "upvote"
    with col2:
        if st.button("👎 Dislike", key="downvote_btn", use_container_width=True):
            st.session_state["feedback_type"] = "downvote"
    with col3:
        if st.button("⚠️ Report this", key="report_btn", use_container_width=True):
            st.session_state["feedback_type"] = "report"

    feedback_type = st.session_state.get("feedback_type", None)
    if feedback_type == "upvote":
        st.success("Thank you for your feedback! 👍")
    elif feedback_type == "downvote":
        st.warning("Thank you for your feedback! 👎")
    elif feedback_type == "report":
        st.info("Thank you for reporting. Please provide more details below.")
        st.text_area("Additional details about why this prediction might be incorrect:", key="report_details", height=100)
        if st.button("Submit Report", key="submit_report_btn"):
            st.success("Report submitted successfully!")
            st.session_state["feedback_type"] = None

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Real Time URL Extraction
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

# Analyze Text 
def analyze_text(text):
    if not text:
        return
    
    with st.spinner("Analyzing..."):
        # Tokenize and get prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get prediction scores
        fake_score = predictions[0][1].item()
        real_score = predictions[0][0].item()
        
        # Display results
        st.subheader("Analysis Results")
        
        # Create a progress bar for visualization
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fake News Probability", f"{fake_score:.2%}")
            st.progress(fake_score)
        with col2:
            st.metric("Real News Probability", f"{real_score:.2%}")
            st.progress(real_score)
        
        # Display conclusion with confidence score
        if fake_score > 0.7:
            verdict = f"Fake ({fake_score:.1%})"
            st.error(f"⚠️ This text shows strong indicators of being fake news (Confidence: {fake_score:.1%})")
        elif fake_score > 0.4:
            verdict = f"Uncertain ({fake_score:.1%})"
            st.warning(f"⚠️ This text shows some indicators of being fake news (Confidence: {fake_score:.1%}). Please verify the information.")
        else:
            verdict = f"Real ({real_score:.1%})"
            st.success(f"✅ This text appears to be legitimate news content (Confidence: {real_score:.1%})")
        
        # Add detailed explanation section
        st.markdown("### 📊 Analysis Summary")
        
        if fake_score > 0.7:
            st.markdown("""
            **Why this might be fake news:**
            - The content shows strong indicators of misinformation
            - The language patterns are consistent with known fake news characteristics
            - The content may contain exaggerated claims or emotional manipulation
            - The information may lack credible sources or verification
            
            **Recommendations:**
            - Cross-reference with reliable news sources
            - Check for official statements or press releases
            - Verify the information with fact-checking websites
            - Be cautious about sharing this content
            """)
        elif fake_score > 0.4:
            st.markdown("""
            **Potential concerns:**
            - Some elements in the content raise questions about its authenticity
            - The information may need additional verification
            - There might be mixed signals in the content's credibility
            
            **Recommendations:**
            - Verify the information with multiple sources
            - Check for official confirmation
            - Look for supporting evidence
            - Exercise caution before sharing
            """)
        else:
            st.markdown("""
            **Why this appears to be legitimate:**
            - The content shows characteristics of reliable news reporting
            - The language patterns are consistent with factual reporting
            - The information appears to be well-structured and verifiable
            
            **Still recommended:**
            - Always verify information from multiple sources
            - Check the publication date and source credibility
            - Look for official statements or documentation
            """)
        
        # Show report options if fake news is detected
        if fake_score > 0.4:
            st.markdown("---")
            show_report_options()

        # Testing purposes only
        #st.markdown("---")
        #show_report_options()
        
        # Show feedback popup only after analysis
        st.session_state['show_feedback_popup'] = True

def show_feedback_popup(text_analyzed, prediction_result):
    # Custom CSS for floating button and popup
    st.markdown('''
        <style>
        .feedback-fab {
            position: fixed;
            bottom: 32px;
            right: 32px;
            z-index: 9999;
        }
        .feedback-popup {
            position: fixed;
            bottom: 90px;
            right: 32px;
            width: 320px;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.18);
            padding: 24px 18px 18px 18px;
            z-index: 10000;
            border: 1px solid #eee;
        }
        .feedback-popup-close {
            position: absolute;
            top: 8px;
            right: 12px;
            font-size: 18px;
            color: #888;
            cursor: pointer;
        }
        </style>
    ''', unsafe_allow_html=True)

    # Floating Action Button (uses Streamlit button)
    if not st.session_state.get('show_feedback_popup', False):
        feedback_btn_placeholder = st.empty()
        with feedback_btn_placeholder.container():
            st.markdown('<div class="feedback-fab">', unsafe_allow_html=True)
            if st.button("💬 Give Feedback", key="open_feedback_popup_btn", help="Give Feedback"):
                st.session_state['show_feedback_popup'] = True
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Feedback Pop-up (Streamlit widgets)
        popup_placeholder = st.empty()
        with popup_placeholder.container():
            st.markdown('<div class="feedback-popup">', unsafe_allow_html=True)
            close = st.button("×", key="close_feedback_popup_btn", help="Close", use_container_width=False)
            st.markdown('<h4 style="margin-top:0">Feedback</h4>', unsafe_allow_html=True)
            feedback = st.radio("How do you rate this prediction?", ["👍 Like", "👎 Dislike", "⚠️ Report"], horizontal=True)
            details = ""
            if feedback == "⚠️ Report":
                details = st.text_area("Report details (optional)", key="popup_report_details", height=60)
            submit = st.button("Submit Feedback", key="submit_feedback_popup_btn")
            st.markdown('</div>', unsafe_allow_html=True)
            if close:
                st.session_state['show_feedback_popup'] = False
                popup_placeholder.empty()
            if submit:
                if feedback == "👍 Like":
                    feedback_db.insert_feedback("upvote", "", text_analyzed, prediction_result)
                    st.success("Thank you for your feedback! 👍")
                elif feedback == "👎 Dislike":
                    feedback_db.insert_feedback("downvote", "", text_analyzed, prediction_result)
                    st.warning("Thank you for your feedback! 👎")
                elif feedback == "⚠️ Report":
                    feedback_db.insert_feedback("report", details, text_analyzed, prediction_result)
                    st.info("Report submitted successfully!")
                time.sleep(2)
                st.session_state['show_feedback_popup'] = False
                popup_placeholder.empty()
                st.experimental_rerun()

try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🔗 URL Input", "📄 PDF Upload"])

# Text Input Tab
with tab1:
    st.markdown("### Paste your news article here")
    text_input = st.text_area("Enter the news text to analyze:", height=200)
    if st.button("Analyze Text", key="analyze_text"):
        if text_input:
            analyze_text(text_input)
        else:
            st.warning("Please enter some text to analyze.")

# URL Input Tab
with tab2:
    st.markdown("### Enter the URL of the news article")
    url_input = st.text_input("Enter the URL:")
    if st.button("Analyze URL", key="analyze_url"):
        if url_input:
            text = extract_text_from_url(url_input)
            if text:
                analyze_text(text)
        else:
            st.warning("Please enter a URL to analyze.")

# PDF Upload Tab
with tab3:
    st.markdown("### Upload a PDF file")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Analyze PDF", key="analyze_pdf"):
        if pdf_file:
            text = extract_text_from_pdf(pdf_file)
            if text:
                analyze_text(text)
        else:
            st.warning("Please upload a PDF file to analyze.")

# --- Feedback Section (Bottom of Page, No Widget, No Close Button) ---

# Only show after analysis
if st.session_state.get('show_feedback_popup', False):
    st.markdown('---')
    st.markdown('## Feedback')
    feedback = st.radio("How do you rate this prediction?", ["👍 Like", "👎 Dislike", "⚠️ Report"], horizontal=True)
    details = ""
    if feedback == "⚠️ Report":
        details = st.text_area("Report details (optional)", key="bottom_report_details", height=60)
    submit = st.button("Submit Feedback", key="submit_feedback_bottom_btn")
    if submit:
        # Save feedback to DB here (you can pass the analyzed text and prediction as needed)
        st.success("Thank you for your feedback!")
        time.sleep(2)
        st.session_state['show_feedback_popup'] = False
        st.experimental_rerun()


