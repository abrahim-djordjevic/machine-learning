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

#initial parameters
k_list = list(range(1, 2000))
sse_list = [0] * len(k_list)

#calculating SSE for each value of k
for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_tsne)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)
        sse += np.linalg.norm(data_tsne[cluster_i]-centroids[i])
    print('k={}, SSE={}'.format(k, sse))
    sse_list[k_ind] = sse

#plotting SSE against k
plt.plot(k_list, sse_list)
plt.show()
