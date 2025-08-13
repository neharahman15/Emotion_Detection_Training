import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Load your CSV file (make sure it's in the same folder)
df = pd.read_csv('EmotionDetection (2).csv')

# Fix for case-sensitive column name
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['Emotion'], test_size=0.2, random_state=42)

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])

print("Training started...")
model.fit(X_train, y_train)
print("Training completed.")

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the trained model to a file in the same folder
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
