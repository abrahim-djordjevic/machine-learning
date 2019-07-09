import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction import stop_words
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import names

#define inpuit dsta
Categories_3 = ['talk.religion.misc','comp.graphics','sci.space']
Groups_3 = fetch_20newsgroups(categories=Categories_3)

def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True

porter_stemmer = PorterStemmer()
all_names = set(names.words())
count_vector_sw = CountVectorizer(stop_words='english', max_features =500)
data_cleaned = []
for doc in Groups_3.data:
	doc_cleaned = ' '.join(porter_stemmer.stem(word) for word in doc.split()
						if is_letter_only(word) and word not in all_names )
	data_cleaned.append(doc_cleaned)
data_cleaned_count = count_vector_sw.fit_transform(Groups_3.data)


#TSNE
tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
data_tsne = tsne_model.fit_transform(data_cleaned_count.toarray())
plt.scatter(data_tsne[:,0],data_tsne[:,1], c=Groups_3.target)
plt.show()
