import pandas as pd
import gensim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import operator
import re
import spacy

circulars = pd.read_csv('Test_File.csv')
circulars.head()

from spacy.lang.en.stop_words import STOP_WORDS

spacy_nlp = spacy.load('en_core_web_sm')

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#function for data cleaning and processing
#This can be further enhanced by adding / removing reg-exps as desired.

def spacy_tokenizer(sentence):
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens

# Commented out IPython magic to ensure Python compatibility.
print ('Cleaning and Tokenizing...')
circulars['Content_tokenized'] = circulars['Content'].map(lambda x: spacy_tokenizer(x))

circulars.head()

content_tokenized = circulars['Content_tokenized']
content_tokenized[0:5]

# Commented out IPython magic to ensure Python compatibility.
from gensim import corpora

#creating term dictionary
dictionary = corpora.Dictionary(content_tokenized)

#filter out terms which occurs in less than 4 documents and more than 20% of the documents.
#NOTE: Since we have smaller dataset, we will keep this commented for now.

#dictionary.filter_extremes(no_below=4, no_above=0.2)

#list of few which which can be further removed
stoplist = set('hello and if this can would should could tell ask stop come go')
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

#print(dictionary)

#print top 50 items from the dictionary with their unique token-id
dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]
#print (dict_tokens)

corpus = [dictionary.doc2bow(desc) for desc in content_tokenized]

word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]

#print(word_frequencies)

#print(corpus)

# Commented out IPython magic to ensure Python compatibility.
circular_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
circular_lsi_model = gensim.models.LsiModel(circular_tfidf_model[corpus], id2word=dictionary, num_topics=300)

# Commented out IPython magic to ensure Python compatibility.
gensim.corpora.MmCorpus.serialize('circular_tfidf_model_mm', circular_tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('circular_lsi_model_mm',circular_lsi_model[circular_tfidf_model[corpus]])

circular_tfidf_corpus = gensim.corpora.MmCorpus('circular_tfidf_model_mm')
circular_lsi_corpus = gensim.corpora.MmCorpus('circular_lsi_model_mm')

# Commented out IPython magic to ensure Python compatibility.
from gensim.similarities import MatrixSimilarity

circular_index = MatrixSimilarity(circular_lsi_corpus, num_features = circular_lsi_corpus.num_terms)

print(circular_index)

from operator import itemgetter

def search_similar_circulars(search_term):

    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = circular_tfidf_model[query_bow]
    query_lsi = circular_lsi_model[query_tfidf]

    circular_index.num_best = 3

    circulars_list = circular_index[query_lsi]

    circulars_list.sort(key=itemgetter(1), reverse=True)
    circulars_names = {}

    for j, relevance in enumerate(circulars_list):

        circulars_names[j] ={
                'Relevance': round((relevance[1] * 100),2),
                'File Name': circulars['File Name'][relevance[0]],
                'Content': circulars['Content'][relevance[0]]
            }

        if j == (circular_index.num_best-1):
            break

    return circulars_names