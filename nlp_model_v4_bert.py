# Import necessary libraries
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the data
data = pd.read_excel(r"complaints_v2.xlsx")

label_map = {'Debt collection': 0, 'Vehicle loan or lease': 1, 'Mortgage': 2, 'Credit card': 3, 'Other financial service': 4}
data['label'] = data['Product'].apply(lambda x: label_map[x])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Consumer'], data['label'], test_size=0.2, random_state=42)

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
X_train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
X_test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

# Convert the labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Create the BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
history = model.fit(x=np.array(X_train_encodings['input_ids']), y=y_train_onehot, validation_data=(np.array(X_test_encodings['input_ids']), y_test_onehot), batch_size=32, epochs=3)

# Evaluate the performance of the model

y_pred = np.argmax(model.predict(np.array(X_test_encodings['input_ids'])).logits.reshape(-1,5), axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# save the model
model.save_pretrained("model")