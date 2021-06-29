import pandas as pd
import gensim
import backend

circulars = pd.read_csv('Test_File.csv')

circular_tfidf_corpus = gensim.corpora.MmCorpus('circular_tfidf_model_mm')
circular_lsi_corpus = gensim.corpora.MmCorpus('circular_lsi_model_mm')

print(circular_tfidf_corpus)
print(circular_lsi_corpus)

# Commented out IPython magic to ensure Python compatibility.
from gensim.similarities import MatrixSimilarity

circular_index = MatrixSimilarity(circular_lsi_corpus, num_features = circular_lsi_corpus.num_terms)

print(circular_index)

from operator import itemgetter

def search_similar_circulars(search_term):

    query_bow = gensim.corpora.Dictionary.doc2bow(backend.spacy_tokenizer(search_term))
    query_tfidf = backend.circular_tfidf_model[query_bow]
    query_lsi = backend.circular_lsi_model[query_tfidf]

    circular_index.num_best = 3

    circulars_list = circular_index[query_lsi]

    circulars_list.sort(key=itemgetter(1), reverse=True)
    circulars_names = []

    for j, relevance in enumerate(circulars_list):

        circulars_names.append (
            {
                'Relevance': round((relevance[1] * 100),2),
                'File Name': circulars['File Name'][relevance[0]],
                'Content': circulars['Content'][relevance[0]]
            }

        )
        if j == (circular_index.num_best-1):
            break

    return pd.DataFrame(circulars_names, columns=['Relevance','File Name','Content']).to_json()