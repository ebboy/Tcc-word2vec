from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import smart_open
import gensim
from gensim.summarization.textcleaner import split_sentences
from gensim.parsing.preprocessing import remove_stopwords
def getTokens(data):
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(data)

    print(count_vect.vocabulary_)
    print(type(X))
    print(X.toarray())
    return count_vect

def getTfIdf(data, modo):
    tfIdf = TfidfTransformer(use_idf=modo).fit(data)
    return tfIdf.fit_transform(data)

def tokenizeWords(data, modo):
    import spacy
    spacy_nlp = spacy.load('en_core_web_sm')
    doc = spacy_nlp(data)
    if(modo == 'w'):
        return [token.text for token in doc]
    elif(modo == 's'):
        return [sent.string.strip() for sent in doc.sents]

def readCorpus(fname, tokens_only=False, mode='s'):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if(mode == 's'):
                # Treinar e adicionar tags
                if(tokens_only):
                    yield split_sentences(remove_stopwords(line))
                else:
                    yield gensim.models.doc2vec.TaggedDocument(split_sentences(remove_stopwords(line)), [i])
            elif(mode == 'w'):
                if(tokens_only):
                    yield gensim.utils.simple_preprocess(line)
                else:
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(remove_stopwords(line)), [i])
