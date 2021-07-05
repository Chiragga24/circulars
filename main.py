import pandas as pd
import gensim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import operator
import re
import spacy

circulars = pd.read_csv('Test_File.csv')

from spacy.lang.en.stop_words import STOP_WORDS

spacy_nlp = spacy.load('en_core_web_sm')

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

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

dictionary = gensim.corpora.Dictionary.load('dictionary.dict')
circular_tfidf_model = gensim.models.TfidfModel.load('circular_tfidf_model')
circular_lsi_model = gensim.models.LsiModel.load('circular_lsi_model')

circular_tfidf_corpus = gensim.corpora.MmCorpus('circular_tfidf_model_mm')
circular_lsi_corpus = gensim.corpora.MmCorpus('circular_lsi_model_mm')

# Commented out IPython magic to ensure Python compatibility.
from gensim.similarities import MatrixSimilarity

circular_index = MatrixSimilarity(circular_lsi_corpus, num_features = circular_lsi_corpus.num_terms)

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
                'URI': circulars['URI'][relevance[0]]
            }

        if j == (circular_index.num_best-1):
            break
    if(not circulars_names):
        return "No result found!"
    return circulars_names