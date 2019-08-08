#importing libraries and packages
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction import stop_words
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import names
from collections import Counter
#define inpuit dsta
Categories_6 = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','sci.crypt','sci.electronics']
Groups_6 = fetch_20newsgroups(categories=Categories_6)

#defining a function for detecting words
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True

#text processing getting that removes 'words' that contain numbers, stemming and stop words
lemmatizer = WordNetLemmatizer()
all_names = set(names.words())
count_vector_sw = CountVectorizer(stop_words='english', max_features =500)
data_cleaned = []
for doc in Groups_6.data:
	doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split()
						if is_letter_only(word) and word not in all_names )
	data_cleaned.append(doc_cleaned)
data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)

#TSNE
tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
data_tsne = tsne_model.fit_transform(data_cleaned_count.toarray())


kmeans = KMeans(n_clusters = 6, random_state = 42)
kmeans.fit(data_tsne)
clusters =kmeans.labels_
print(Counter(clusters))
