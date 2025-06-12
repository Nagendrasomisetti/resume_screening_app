import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Sample data: resumes and labels (1 = good, 0 = bad)
data = {
    "resume": [
        "Experienced Python developer with ML background",
        "Fresh graduate with no experience",
        "Data scientist with 5 years experience in AI",
        "Good at cricket and cooking, no IT skills"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Split and vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume"])
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model trained and saved.")
