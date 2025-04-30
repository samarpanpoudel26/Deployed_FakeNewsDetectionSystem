"""
Fake News Detection Web Application

This module implements a Streamlit-based web interface for the fake news detection system.
It provides a user interface for text input, analysis, and viewing detection history.
"""

import pandas as pd
import streamlit as st
import json
import os
import time
from datetime import datetime
from model1 import user_input

# Constants
LOG_FILE = 'user_logs.json'
CSS_STYLES = """
    <style>
    html, body, .stApp {
        height: 100%;
        background: linear-gradient(135deg, #e0f7fa, #80cbc4);
        color: #00332d;
        font-family: 'Segoe UI', sans-serif;
        margin: 0;
        padding: 0;
    }
    h1, h2, h3, h4 {
        color: #004d40;
    }
    .stTextInput > div > div > input {
        background-color: #ffffffcc;
        color: #00332d;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        border: none;
    }
    .stButton > button {
        background-color: #00796b;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        border: none;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #004d40;
        transform: scale(1.02);
    }
    .stSidebar {
        background-color: #004d40 !important;
    }
    .stSidebar h2 {
        color: white !important;
    }
    .stSidebar .css-1v0mbdj {
        color: white !important;
    }
    .stSidebar .stMarkdown {
        color: white !important;
    }
    .stDataFrame {
        background-color: #ffffff33;
    }
    </style>
"""

class FakeNewsUI:
    """
    A class to handle the Streamlit user interface for fake news detection.
    
    Attributes:
        logs (list): List of detection history entries
    """
    
    def __init__(self):
        """Initialize the UI components and load existing logs."""
        self.setup_page_style()
        self.logs = self.load_logs()
        
    def setup_page_style(self):
        """Set up the page styling and CSS."""
        st.markdown(CSS_STYLES, unsafe_allow_html=True)
        
    def load_logs(self) -> list:
        """
        Load existing detection history from JSON file.
        
        Returns:
            list: List of detection history entries
        """
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as file:
                return json.load(file)
        return []
    
    def save_prediction(self, text: str, result: str):
        """
        Save prediction result to logs.
        
        Args:
            text (str): Input news text
            result (str): Prediction result ('Real' or 'Fake')
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "News": text,
            "prediction": result
        }
        self.logs.append(entry)
        with open(LOG_FILE, 'w') as file:
            json.dump(self.logs, file, indent=4)
    
    def display_result(self, prediction: int):
        """
        Display the prediction result with appropriate styling.
        
        Args:
            prediction (int): Model prediction (0 for Fake, 1 for Real)
        """
        if prediction == 0:
            result = "Fake"
            st.markdown(
                "<div style='background-color:#FFDDDD;padding:10px;border-radius:5px;'>"
                "❌ This news appears to be <b>Fake</b></div>",
                unsafe_allow_html=True
            )
        else:
            result = "Real"
            st.markdown(
                "<div style='background-color:#DDFFDD;padding:10px;border-radius:5px;'>"
                "🟢 This news appears to be <b>Real</b></div>",
                unsafe_allow_html=True
            )
        return result
    
    def show_history(self):
        """Display the detection history in a dataframe."""
        with open(LOG_FILE, 'r') as file:
            logs = json.load(file)
        df = pd.DataFrame(logs)
        st.dataframe(df)
    
    def run(self):
        """Run the main application loop."""
        # Set up header
        st.markdown(
            "<h1 style='text-align: center;'>📰 Fake News Detector</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #004d40;'>"
            "Enter a news headline or body text to analyze.</p>",
            unsafe_allow_html=True
        )
        
        # Get user input
        news_text = st.text_input("")
        
        # Handle analysis
        if st.button("Analyze", key="1"):
            if not news_text:
                st.warning("⚠️ Please enter news text before analyzing")
            else:
                with st.spinner("Analyzing..."):
                    time.sleep(2)
                prediction = user_input(news_text)
                result = self.display_result(prediction)
                self.save_prediction(news_text, result)
        
        # Show history if requested
        if st.checkbox("View History"):
            self.show_history()
        
        # Display sidebar information
        self.display_sidebar()
    
    def display_sidebar(self):
        """Display project information in the sidebar."""
        st.sidebar.markdown("""
            <div style='
                background-color: #00695c;
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
            '>
                <h3>ℹ️ About This Project</h3>
                <p>This model was developed as part of the <b>CS196 Final Project</b>.</p>
                <ul>
                    <li>It detects whether a news input is <i>Real</i> or <i>Fake</i>.</li>
                    <li>⚠️ The model is not 100% accurate.</li>
                    <li>Based on limited training data.</li>
                    <li>Use for <b>informational purposes</b> only.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = FakeNewsUI()
    app.run()



    

