# %%
"""
Fake News Detection Model

This module implements a machine learning model for detecting fake news using Natural Language Processing (NLP)
techniques and a Multinomial Naive Bayes classifier.
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# %% Data Loading and Preprocessing
class TextPreprocessor:
    """
    A class for preprocessing text data for fake news detection.
    
    Methods:
        preprocess_text(text: str) -> str: Cleans and preprocesses input text
    """
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocesses text by performing the following steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove punctuation
        4. Remove numbers
        5. Remove stopwords
        6. Tokenize and join words
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Join tokens back into text
        cleaned_text = " ".join(tokens)
        
        return cleaned_text

# %% Model Class
class FakeNewsDetector:
    """
    A class for detecting fake news using machine learning.
    
    Attributes:
        vectorizer (CountVectorizer): Text vectorization tool
        model (MultinomialNB): Trained Naive Bayes classifier
    """
    
    def __init__(self):
        """Initialize the FakeNewsDetector with vectorizer and model."""
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.preprocessor = TextPreprocessor()
    
    def train(self, data_path: str):
        """
        Train the fake news detection model.
        
        Args:
            data_path (str): Path to the training data file
        """
        # Load and preprocess data
        dataset = pd.read_csv(data_path)
        dataset.drop(columns=["URLs", "Headline"], axis=1, inplace=True)
        dataset.dropna(inplace=True)
        dataset = dataset.sample(frac=1)  # Shuffle data
        
        # Preprocess text
        dataset["Body"] = dataset["Body"].apply(self.preprocessor.preprocess_text)
        
        # Vectorize text and prepare features
        X = self.vectorizer.fit_transform(dataset["Body"])
        y = dataset[["Label"]]
        
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate and store accuracy scores
        self.test_accuracy = self.model.score(X_test, y_test) * 100
        self.train_accuracy = self.model.score(X_train, y_train) * 100
    
    def predict(self, text: str) -> int:
        """
        Predict whether input text is fake news or not.
        
        Args:
            text (str): Input news text
            
        Returns:
            int: 1 for real news, 0 for fake news
        """
        # Preprocess input text
        cleaned_text = self.preprocessor.preprocess_text(text)
        
        # Transform text and predict
        transformed_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(transformed_text)[0]
        
        return prediction

# Initialize and train model
detector = FakeNewsDetector()
detector.train("data.xls")

def user_input(user_text: str) -> int:
    """
    Process user input and return prediction.
    
    Args:
        user_text (str): News text to classify
        
    Returns:
        int: 1 for real news, 0 for fake news
    """
    prediction = detector.predict(user_text)
    
    if prediction == 1:
        print("Not a fake News")
    else:
        print("Fake news")
    
    return prediction



