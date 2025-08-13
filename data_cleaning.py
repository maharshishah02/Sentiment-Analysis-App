import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# --- Download NLTK data if not already present ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """Cleans the input text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower() # Remove non-alphabetic chars
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Main script ---
if __name__ == "__main__":
    print("Starting data cleaning...")

    # Load raw data
    df = pd.read_csv('data/IMDB Dataset.csv')

    # Apply the cleaning function
    df['cleaned_review'] = df['review'].apply(clean_text)

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the cleaned data to a pickle file 💾
    df[['cleaned_review', 'sentiment']].to_pickle('data/cleaned_reviews.pkl')
    
    print("Data cleaning complete. Cleaned data saved to 'data/cleaned_reviews.pkl'")