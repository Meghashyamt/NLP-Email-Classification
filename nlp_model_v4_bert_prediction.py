import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification


loaded_model = TFBertForSequenceClassification.from_pretrained("model")

#Load the test data
test_data = pd.read_excel("test_emails.xlsx")

#Tokenize the text data
test_encodings = tokenizer(test_data['body'].tolist(), truncation=True, padding=True)

#Get the predicted labels
test_pred = np.argmax(loaded_model.predict(np.array(test_encodings['input_ids'])).logits.reshape(-1, 3), axis=1)

#Add predicted labels to the test data
test_data['predicted_label'] = test_pred

#Save the results
test_data.to_excel("predicted_labels.xlsx", index=False)