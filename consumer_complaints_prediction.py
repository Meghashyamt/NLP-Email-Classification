#from consumer_complaints import *
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

model = load('model_v2.joblib')

# Load test data from excel
test_df = pd.read_excel('test_data.xlsx')

# Make predictions on the test data
#predictions = model.predict(test_df)
#text_features = tfidf.transform(test_df)
#predictions = model.predict(text_features)
#text="I got a call from somebody asking me to resolve pending debt that I owe to your company. However, as of today I don’t have anything left. But still your team keep harassing me."
#testdata=np.array(text).reshape(1,-1)
tfidf=load('tfidf_v2.joblib')
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#tfidf.fit_transform(test_df['body'])

id_to_category={0: 'Credit card', 1: 'Debt collection', 2: 'Mortgage', 3: 'Vehicle loan or lease', 4: 'Unclassified', 5: 'HR', 6: 'Finance', 7: 'IT Support', 8: 'People'}
text_features = tfidf.transform(test_df['body'])
predictions=model.predict(text_features)

print(predictions)
"""for text, predicted in zip(test_df, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")"""
# Add the predictions as a new column to the test data
#test_df['predictions'] = id_to_category[predictions]
#test_df['predicted_category'] = predictions.apply(lambda x: id_to_category[x])

predicted_categories = [id_to_category[prediction] for prediction in predictions]
test_df['predicted_category'] = predicted_categories

# Save the test data with predictions to excel
test_df.to_excel('test_data_with_predictions.xlsx', index=False)