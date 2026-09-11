import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# 1. Load the dataset
# -----------------------------------

data = pd.read_csv("dataset/complaints.csv")


# -----------------------------------
# 2. Remove duplicate rows
# -----------------------------------

data = data.drop_duplicates()


# -----------------------------------
# 3. Separate input and output
# -----------------------------------

X = data["complaint"]
y = data["category"]


# -----------------------------------
# 4. Split the dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 5. Convert text into numbers
# -----------------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# 6. Create the ML model
# -----------------------------------

model = LogisticRegression(max_iter=1000)


# -----------------------------------
# 7. Train the model
# -----------------------------------

model.fit(X_train_tfidf, y_train)
# Save the category model
joblib.dump(model, "models/category_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/category_vectorizer.pkl")

print("\nCategory model saved successfully!")

# -----------------------------------
# 8. Make predictions
# -----------------------------------

y_pred = model.predict(X_test_tfidf)


# -----------------------------------
# 9. Calculate accuracy
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("========== CATEGORY MODEL ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# 10. Test with a new complaint
# -----------------------------------

new_complaint = [
    "Money was deducted from my account but my payment failed"
]

new_complaint_tfidf = vectorizer.transform(new_complaint)

prediction = model.predict(new_complaint_tfidf)

probabilities = model.predict_proba(new_complaint_tfidf)

confidence = probabilities.max() * 100


print("\nNew complaint:")
print(new_complaint[0])

print("\nPredicted category:")
print(prediction[0])

print("Confidence:")
print(round(confidence, 2), "%")