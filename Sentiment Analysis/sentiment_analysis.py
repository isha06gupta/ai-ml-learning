import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# ==========
# LOAD DATA
# ==========

train_data = pd.read_csv("train.csv", encoding="latin-1") #unicode error avoid karne ke liye latin encoding(byte->text) because it has special characters. 
test_data  = pd.read_csv("test.csv",  encoding="latin-1")

# Reduce training size
if len(train_data) > 20000:
    train_data = train_data.sample(n=20000, random_state=42)
    #If the training dataset is very large, randomly select 20,000 rows from it.
    #random_state=42: Every time you run the program, the same 20,000 tweets are selected.

# =========================
# SELECT REQUIRED COLUMNS
# =========================

train_data = train_data[['text', 'sentiment']]
test_data  = test_data[['text', 'sentiment']]

# ============
# CLEAN LABELS
# =============

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
#sentiments-> numbers , mapping

train_data['sentiment'] = train_data['sentiment'].map(label_map) 
test_data['sentiment']  = test_data['sentiment'].map(label_map)

train_data = train_data.dropna(subset=['sentiment', 'text'])
#remove row where sentiment or text are missing
test_data  = test_data.dropna(subset=['sentiment', 'text'])

train_data['sentiment'] = train_data['sentiment'].astype(int)
#sentiment column->integer type(0,1,2) for ML Model
test_data['sentiment']  = test_data['sentiment'].astype(int)


# ===========
# TEXT CLEAN 
# ===========

train_data['text'] = train_data['text'].astype(str).str.lower()
#tweet text to lowercase strings, for consistency 
test_data['text']  = test_data['text'].astype(str).str.lower()

train_data.loc[train_data['text'].str.strip() == "", 'text'] = "empty"
#empty tweet -> "empty"
test_data.loc[test_data['text'].str.strip() == "", 'text'] = "empty"

#these changes for TF-IDF 

# ========
# X AND Y
# ========

X_train = train_data['text']
Y_train = train_data['sentiment']

X_test = test_data['text']
Y_test = test_data['sentiment']

# =======
# TF-IDF
# =======
#tweet text -> numerical features
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=2000,
    token_pattern=r"(?u)\b\w+\b"
)
#remove stopwords,limit to 2000 features, regex to ensure valid tokens

X_train_tfidf = tfidf.fit_transform(X_train)
#learns vocab and tf-idf weight from training data , training text->numerical vectors

X_test_tfidf  = tfidf.transform(X_test)
#test data ->tf-idf vectors using same vocab

# ===================
# SAVE X AND Y FILES
# ===================

pd.DataFrame(X_train_tfidf.toarray()).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test_tfidf.toarray()).to_csv("X_test.csv", index=False)
#tf-idf feature matric -> CSV file
#toarray() sparse format->normal table to save as CSV

Y_train.to_csv("Y_train.csv", index=False)
Y_test.to_csv("Y_test.csv", index=False)

# =======
# MODEL
# =======

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, Y_train)

Y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

