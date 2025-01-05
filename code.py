import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your datasets (assuming CSV format for simplicity)
train_data = pd.read_csv('unfair_tos_train.csv')
validation_data = pd.read_csv('unfair_tos_validation.csv')
test_data = pd.read_csv('unfair_tos_test.csv')

# Assuming the last column is the label and the rest are text features
X_train = train_data.iloc[:, :-1]  # Features (text)
y_train = train_data.iloc[:, -1]   # Label

X_validation = validation_data.iloc[:, :-1]  # Features (text)
y_validation = validation_data.iloc[:, -1]   # Label

X_test = test_data.iloc[:, :-1]  # Features (text)
y_test = test_data.iloc[:, -1]   # Label

# Combine all labels from train, validation, and test datasets
all_labels = pd.concat([y_train, y_validation, y_test])

# Initialize and fit the LabelEncoder on all labels
encoder = LabelEncoder()
encoder.fit(all_labels)

# Transform labels using the fitted encoder
y_train = encoder.transform(y_train)
y_validation = encoder.transform(y_validation)
y_test = encoder.transform(y_test)

# Convert the text features to numeric values using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data, transform the validation and test data
X_train_tfidf = vectorizer.fit_transform(X_train.iloc[:, 0])  # Assuming text is in the first column
X_validation_tfidf = vectorizer.transform(X_validation.iloc[:, 0])  # Validation text
X_test_tfidf = vectorizer.transform(X_test.iloc[:, 0])  # Test text

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Validation set performance
y_val_pred = model.predict(X_validation_tfidf)
print("\nValidation Accuracy:", accuracy_score(y_validation, y_val_pred))
print("Validation Classification Report:")
print(classification_report(y_validation, y_val_pred))

# Test set performance
y_test_pred = model.predict(X_test_tfidf)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# Function to take user input and rank relevant documents
def rank_documents(query, documents, vectorizer, top_n=5):
    # Convert the input query to TF-IDF features
    query_tfidf = vectorizer.transform([query])
    
    # Calculate cosine similarity between the query and the documents
    cosine_sim = cosine_similarity(query_tfidf, documents)
    
    # Rank the documents based on cosine similarity score
    ranked_indices = cosine_sim.argsort()[0][-top_n:][::-1]
    
    # Print the most relevant documents
    print("\nTop 5 most relevant documents to the query:")
    for idx in ranked_indices:
        print(f"Document Index: {idx}, Similarity Score: {cosine_sim[0][idx]:.4f}")
        print(documents[idx])  # Print the relevant document text
        print("="*80)

# Example: User provides a query to rank relevant documents from the training set
query = input("Enter a query (text) to find relevant documents: ")

# Rank documents from the training set
rank_documents(query, X_train_tfidf, vectorizer)
