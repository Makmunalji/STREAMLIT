import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')

# Load pre-trained model and vectorizer
model = joblib.load('logistic_model')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app
def main():
    # Set page title and description
    st.set_page_config(page_title="Text Preprocessing and Classification App")
    
    # Title of the app
    st.title("Text Preprocessing and Classification App")
    st.write("Aplikasi ini merupakan menggunakan preprocesses text dan membuat prediksi menggunakan pre-trained Logistic Regression Model.")
    
    # Input for news text
    user_input = st.text_input("Masukkan text:")
    
    if user_input:
        # Preprocess the input text
        cleaned_text = preprocess_text(user_input)
        
        # Transform the cleaned text using the pre-trained vectorizer
        text_features = vectorizer.transform([cleaned_text])
        
        # Predict the class using the pre-trained model
        prediction = model.predict(text_features)
        
        # Mapping predictions to categories
        categories = {0: "Bola", 1: "Health"}
        result = categories.get(prediction[0], "Unknown")
        
        # Display the result
        st.write(f"Prediksi Berita: {result}")
    else:
        st.write("Masukkan teks untuk prediksi.")
main()
