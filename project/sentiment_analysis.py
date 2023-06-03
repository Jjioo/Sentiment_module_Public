import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the saved models
RandomForest_model = joblib.load('RandomForest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
model = joblib.load('model.pkl')
naive_model = joblib.load('naive_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')

# Load the saved CountVectorizer
vectorizer = joblib.load('vectorizer.pkl')

new_sentence = "The GeoSolutions technology will leverage Benefon's GPS solutions by providing Location Based Search Technology, a Communities Platform, location relevant multimedia content and a new and powerful commercial model."

# Transform the new sentence using the loaded vectorizer
new_sentence_vectorized = vectorizer.transform([new_sentence])

# Create the VotingClassifier and fit it with the loaded models
voting_classifier = VotingClassifier([("RandomForest", RandomForest_model),
                                      ("SVM", svm_model),
                                      ("Model", model),
                                      ("NaiveBayes", naive_model)])
voting_classifier.fit(X_test, y_test)

# Predict the sentiment for the new sentence using the fitted VotingClassifier
sentiment = voting_classifier.predict(new_sentence_vectorized)

# Decode the predicted sentiment
sentiment_decoded = label_encoder.inverse_transform(sentiment)

# Predict the sentiment for the test data using the fitted VotingClassifier
y_pred = voting_classifier.predict(X_test)

# Decode the predicted sentiment
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_decoded)


