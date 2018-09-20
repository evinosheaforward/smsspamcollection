import io
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora
import io


def read_data(fpath):
    data = {'ham': [], 
            'spam': []
            }
    with io.open(fpath, 'r', encoding='utf8') as infile:
        for line in infile:
            if line.startswith("ham"):
                data['ham'].append(line[4:].rstrip())
            elif line.startswith("spam"):
                data['spam'].append(line[5:].rstrip())
            else:
                print("Line not starting wiht spam or ham found...")

    return data


def capitalized_statistics(data):
    """This data should be the list of sentances of ham or spam, not the dictionary
    """
    total_words = 0
    total_capitalized = 0
    percents = []
    for sentance in data:
        num_words, num_capitalized, percent_capitalized = get_capitalized(sentance)
        total_words += num_words
        total_capitalized += num_capitalized
        percents.append(percent_capitalized)

    ave_capitalized = float(total_capitalized) / total_words

    return ave_capitalized, percents, total_words


def get_capitalized(sentance):
    words = sentance.split()
    
    capitalized = [word for word in words if word[0].isupper() and word != ""]

    num_words = len([word for word in words if word != ""])
    num_capitalized = len(capitalized)
    percent_capitalized = float(num_capitalized) / num_words

    return num_words, num_capitalized, percent_capitalized


def get_punctuation(sentance, punct_marks, numbers=None):
    """This function takes in a sentacen and list of punctuation marks and returns
    a dictionary of the number of occurances of the punctuation mark in the sentance.
    This can also be done cumulatively if "numbers" is passed where numbers is a dictionary
    with the punctuation marks as keys and values of numbers
    """
    if numbers is None:
        numbers = {punct: 0 for punct in punct_marks}

    for punct in punct_marks:
        numbers[punct] += len([char for char in sentance if char == punct]) 

    return numbers


def features(sentance):
    punct_marks = ["!", ":", "'"]
    numbers = get_punctuation(sentance, punct_marks)
    num_words, num_capitalized, percent_capitalized = get_capitalized(sentance)
    feats = [numbers[punctuation_mark] for punctuation_mark in punct_marks]
    feats.extend([num_capitalized, percent_capitalized])
    return feats


def tokenizer(sentance):  
    ne_tree = ne_chunk(pos_tag(word_tokenize(sentance.encode('ascii', 'ignore'))))

    iob_tagged = tree2conlltags(ne_tree)

    words = [i[0] for i in iob_tagged]
    return words 


def analyzer(data):
    pass


def split_train(features_matrix, labels, model=XGBClassifier(), test_size=0.2, return_split=False):


    features_train, features_test, \
                    labels_train, \
                    labels_test = train_test_split(features_matrix, 
                                                    labels, 
                                                    test_size=test_size)

    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)

    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    if not return_split:
        return model
    else:
        return model, features_train, features_test, labels_train, labels_test, labels_pred

                                                  
"""
def term_matrix(words):
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(words)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    return [dictionary.doc2bow(doc) for doc in words]

def read_data(fpath):
    data = {'ham': [], 
            'spam': []
            }
    with io.open(fpath, 'r') as infile:
        for line in infile:
            if line.startswith("ham"):
                data['ham'].append(line[4:].rstrip())
            elif line.startswith("spam"):
                data['spam'].append(line[5:].rstrip())
            else:
                print("Line not starting wiht spam or ham found...")

    return data


def clean(doc):
    lemma = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def num_nouns(sentance):  
    Return number of nouns in sentance reguardless of type

    ne_tree = ne_chunk(pos_tag(word_tokenize(sentance.encode('ascii', 'ignore'))))
    nn = len([0 for i in ne_tree if i[1].startswith('NN')])
    return nn

def most_information(document, vectorizer):
    
    :param document: 
    :rtype: `dict`
    sentances = document.split(".")
    #Maybe better than just spling on .?

    nns = [(sentance, num_nouns(sentance)) for sentance in sentances]

    for sent, nn in nns:
        vect = vectorizer.transform(sentance)
        for scomp, nncomp in nns:
            if sent != scomp:
                continue#magic happens
            else: 
                continue







## wil these be different for the new project?

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        print "fname=", fname
        with open(fname) as pearl:
            text = pearl.read()
            token_dict[f] = text.lower().translate(None, string.punctuation)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())

str = 'all great and precious things are lonely.'
response = tfidf.transform([str])
print response

feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print feature_names[col], ' - ', response[0, col]


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)
"""