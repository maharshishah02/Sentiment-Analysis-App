# 🎬 Movie Review Sentiment Analysis Web App

An end-to-end machine learning project that classifies movie reviews as either **positive** or **negative**. This project uses a Logistic Regression model trained on the IMDB 50K Movie Reviews dataset and is deployed as an interactive web application using Streamlit.

---

## ✨ Features

* **Data Cleaning:** A robust pipeline to clean and preprocess raw text data by removing HTML tags, punctuation, and stop words.
* **Feature Engineering:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into meaningful numerical vectors for the model.
* **ML Model:** A Logistic Regression classifier trained and evaluated for high accuracy and performance.
* **Interactive UI:** A user-friendly web app built with Streamlit that allows users to enter any movie review and get a real-time sentiment prediction.
* **Modular Code:** The project is cleanly structured into separate Python scripts for each stage of the ML lifecycle: data cleaning, feature engineering, model training, and deployment.

---

## 🚀 Demo

Here is a screenshot of the final web application in action:

<img width="1211" height="802" alt="Positive Analysis" src="https://github.com/user-attachments/assets/862ee560-e375-45dd-a4f4-0ade4b0cf015" />

<img width="1171" height="767" alt="Negative Analysis" src="https://github.com/user-attachments/assets/4ef3c62a-4f71-4cef-9ff3-2aee898bd491" />


---

## 📁 Project Structure

---

## 🛠️ How to Run This Project

To run this project locally, please follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/sentiment-analysis-app.git](https://github.com/your-username/sentiment-analysis-app.git)
cd sentiment-analysis-app
```
### 2. Create and Activate a Virtual Environment
# Create a virtual environment
python -m venv ml_env

# Activate it (Windows)
.\ml_env\Scripts\activate

# Activate it (macOS/Linux)
source ml_env/bin/activate

### 3. Install Dependencies
Install all the necessary libraries from the requirements.txt file.

pip install -r requirements.txt

### 4. Download the Dataset
Download the IMDB Dataset of 50K Movie Reviews from Kaggle.
Create a folder named data in the project's root directory.
Place the downloaded IMDB Dataset.csv file inside the data folder.

### 5. Run the ML Pipeline
Execute the scripts in order to clean the data, create features, and train the model.

python 1_data_cleaning.py
python 2_feature_engineering.py
python 3_model_training.py

### 6. Launch the Web App
Start the Streamlit application.

streamlit run app.py

💻 Technologies Used
- Python
- Pandas for data manipulation  
- Scikit-learn for machine learning
- NLTK for text preprocessing
- Streamlit for the web interface
- Matplotlib & Seaborn for visualizations
