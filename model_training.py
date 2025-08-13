import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- Main script ---
if __name__ == "__main__":
    print("Starting model training...")

    # 1. Load the processed data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    tfidf_vectorizer = data['tfidf_vectorizer']

    # 2. Train a Logistic Regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Evaluate the model
    y_pred = log_reg.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification Report (Precision, Recall, F1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 4. Display the Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plotting the confusion matrix for better visualization
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # 5. Save the final model and vectorizer 💾
    final_model_data = {
        'model': log_reg,
        'vectorizer': tfidf_vectorizer
    }
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    with open('models/sentiment_model.pkl', 'wb') as f:
        pickle.dump(final_model_data, f)
        
    print("\nTrained model and vectorizer saved to 'models/sentiment_model.pkl'")