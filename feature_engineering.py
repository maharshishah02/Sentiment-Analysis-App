import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Main script ---
if __name__ == "__main__":
    print("Starting feature engineering...")

    # Load the cleaned data
    df = pd.read_pickle('data/cleaned_reviews.pkl')

    # Encode labels
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

    # Split data
    X = df['cleaned_review']
    y = df['sentiment_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the processed data and the vectorizer
    processed_data = {
        'X_train': X_train_tfidf,
        'X_test': X_test_tfidf,
        'y_train': y_train,
        'y_test': y_test,
        'tfidf_vectorizer': tfidf
    }
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
        
    print("Feature engineering complete. Processed data saved to 'data/processed_data.pkl'")