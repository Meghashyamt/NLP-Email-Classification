# Load the dataset
import pandas as pd
import re
data = pd.read_excel('emails.xlsx')
print(data.head())

# Pre-processing
import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


stop_words = set(stopwords.words('english'))
ps=PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
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

data['body'] = data['body'].apply(preprocess_text)

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['body'])
y = data['label']

# Model Selection
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# Training and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(data)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))

# Prediction
new_email = "schedule a demo"
new_email = preprocess_text(new_email)
new_email_vec = vectorizer.transform([new_email])
print(model.predict(new_email_vec))
