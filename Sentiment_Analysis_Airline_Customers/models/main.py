import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
dataset_path = 'Tweets.csv'
tweets_df = pd.read_csv(dataset_path)

# -------- EDA --------
# Sentiment Distribution (example EDA)
plt.figure(figsize=(8, 6))
sns.countplot(x='airline_sentiment', data=tweets_df, palette='Set2')
plt.title('Distribution of Sentiment Classes')
plt.show()

# 2. Word Cloud for the Most Frequent Words in Tweets
all_words = ' '.join([text for text in tweets_df['text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# 3. Sentiment by Airline
plt.figure(figsize=(10, 6))
sns.countplot(x='airline', hue='airline_sentiment', data=tweets_df, palette='coolwarm')
plt.title('Sentiment Distribution by Airline')
plt.show()

# -------- Preprocessing --------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization and Stopword removal
    return text

# Apply preprocessing to the 'text' column
tweets_df['cleaned_text'] = tweets_df['text'].apply(preprocess_text)

# Prepare the feature (X) and target (y) variables
X = tweets_df['cleaned_text']
y = tweets_df['airline_sentiment']

# Encode target labels (y) into numeric format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Save the TF-IDF vectorizer for use during API deployment
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Dictionary to store model performance
model_performance = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)  # Train the model on encoded labels
    y_pred = model.predict(X_test_tfidf)  # Predict encoded labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)  # Decode predictions back to string format
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)  # Use encoded labels for evaluation
    
    # Generate classification report and confusion matrix using decoded labels
    class_report = classification_report(label_encoder.inverse_transform(y_test), y_pred_decoded)
    conf_matrix = confusion_matrix(label_encoder.inverse_transform(y_test), y_pred_decoded)
    
    # Store model performance
    model_performance[model_name] = {
        'Accuracy': accuracy,
        'Classification Report': class_report,
        'Confusion Matrix': conf_matrix
    }
    
    # Save the model for API deployment
    joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}_model.pkl")

# Display the results
for model_name, performance in model_performance.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {performance['Accuracy']}")
    print(f"Confusion Matrix:\n{performance['Confusion Matrix']}")
    print(f"Classification Report:\n{performance['Classification Report']}")
    print("-" * 50)
