#importing libraries and packages
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
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
Groups_6 = fetch_20newsgroups()

#defining a function for detecting words
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True

#text processing getting that removes 'words' that contain numbers, stemming and stop words
lemmatizer = WordNetLemmatizer()
all_names = set(names.words())
tfid_vector = TfidfVectorizer(stop_words='english', max_features =500)
data_cleaned = []
for doc in Groups_6.data:
	doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split()
						if is_letter_only(word) and word not in all_names )
	data_cleaned.append(doc_cleaned)
data_cleaned_count = tfid_vector.fit_transform(data_cleaned)

#NMF
from sklearn.decomposition import NMF
t = 6
nmf = NMF(n_components=t, random_state=42)
nmf.fit(data_cleaned_count)
nmf.components_
terms = tfid_vector.get_feature_names()
for topic_idx, topic in enumerate(nmf.components_):
    print('Topic():'.format(topic_idx))
    print(' '.join([terms[i] for i in topic.argsort()[-10:]]))
