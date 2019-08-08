import glob
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

reviews, label = [], []
file_name = []
df = pd.DataFrame()

pos_path = 'C:\\Users\\User\\Documents\\review_polarity\\txt_sentoken\\pos\\'
neg_path = 'C:\\Users\\User\\Documents\\review_polarity\\txt_sentoken\\neg\\'

def file_load(path):
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                file_name.append(os.path.join(r, file))
                if path == pos_path:
                    label.append(0)
                elif path == neg_path:
                    label.append(1)
file_load(pos_path)
file_load(neg_path)

for file in file_name:
    with open(file, 'r', encoding='ISO-8859-1') as infile:
        reviews.append(infile.read())

#text preprocessing
def is_letter_only(word):
    return word.isalpha()

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    docs_cleaned = []
    for doc in docs:
        doc = doc.lower()
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if is_letter_only(word)
                      and word not in all_names)
        docs_cleaned.append(doc_cleaned)
    return docs_cleaned

reviews_cleaned = clean_text(reviews)
reviews_cleaned = str(reviews_cleaned)
reviews_cleaned = reviews_cleaned.split()
reviews_cleaned = list(reviews_cleaned)
#feature_extraction
cv = CountVectorizer(stop_words='english', max_features = 2000, max_df = 50, min_df=1)
docs_cv = cv.fit_transform(reviews_cleaned)
terms= cv.get_feature_names()

#splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(terms, label, test_size = 0.2, random_state=42)
term_docs_train = cv.transform(X_train)
term_docs_test = cv.transform(X_test)

#testing
clf = MultinomialNB(alpha=1.0, fit_prior = False)
clf.fit(term_docs_train, Y_train)
prediction = clf.predict(term_docs_test)
#review
accuracy = clf.score(term_docs_test, Y_test)
print(prediction[:2000])
print('accuracy is:{0:.1f}%'.format(accuracy*100))
