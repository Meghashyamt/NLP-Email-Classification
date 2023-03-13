import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer


# Load the dataset
data = pd.read_excel('emails.xlsx')
print(data.head())

# Preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
"""
def preprocess(text):
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text
"""
def preprocess(text):
    # Clean the text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    filtered_tokens = [token for token in tokens if not token in stop_words]
    filtered_tokens = [ps.stem(token) for token in tokens if  token not in stopwords.words('english')]
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if  token not in stopwords.words('english')]
    
    # Return the preprocessed text
    return " ".join(filtered_tokens)

data['text'] = data['body'].apply(preprocess)

# Feature Extraction
#vectorizer = CountVectorizer()
vectorizer=TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])

# Splitting the data
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(data)
# Model Selection and Training
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluation
y_pred = nb_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Prediction
new_text = "demo request"
new_text=preprocess(new_text)
new_text_transformed = vectorizer.transform(new_text)
print(nb_classifier.predict(new_text_transformed))
