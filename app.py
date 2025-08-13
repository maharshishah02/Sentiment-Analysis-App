import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Text Cleaning Function (must be identical to the one used for training) ---
def clean_text(text):
    """Cleans the input text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower() # Remove non-alphabetic chars
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Load Model and Vectorizer ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    with open('models/sentiment_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model()
model = model_data['model']
vectorizer = model_data['vectorizer']

# --- Streamlit App Interface ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.subheader("Enter a movie review to classify it as Positive or Negative")

# User input text area
user_input = st.text_area("Enter your review here:")

# Prediction button
if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Clean the user input
        cleaned_input = clean_text(user_input)
        
        # 2. Vectorize the cleaned input
        input_vectorized = vectorizer.transform([cleaned_input])
        
        # 3. Predict the sentiment
        prediction = model.predict(input_vectorized)
        
        # Display the result
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.success("This review is POSITIVE! 👍")
        else:
            st.error("This review is NEGATIVE. 👎")
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("---")
st.write("Built by an awesome AI/ML Engineer 😉")