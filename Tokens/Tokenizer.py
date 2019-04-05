from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import smart_open
import gensim
from gensim.summarization.textcleaner import split_sentences
from gensim.parsing.preprocessing import remove_stopwords
import nltk
def getVectors(data):
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(data)

    print(count_vect.vocabulary_)
    print(type(X))
    print(X.toarray())
    return count_vect

def getTfIdf(data, mode=False):
    tfIdf = TfidfTransformer(use_idf=mode).fit(data)
    return tfIdf.fit_transform(data)

def tokenizeWords(data, mode='s'):
    import spacy
    spacy_nlp = spacy.load('en_core_web_sm')
    doc = spacy_nlp(data)
    if(mode == 'w'):
        return [token.text for token in doc]
    elif(mode == 's'):
        return [sent.string.strip() for sent in doc.sents]

def readCorpus(fname, tokens_only=False, mode='w'):
    tokens = []
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if(mode == 's'):
                 tokens.append(split_sentences(remove_stopwords(line)))
            else:  # Train text with or without tags
                 tokens.append(gensim.utils.simple_preprocess(remove_stopwords(line)))
    return tokens


# def readCorpus(fname, tokens_only=False, mode='s'):
#     with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
#         for i, line in enumerate(f):
#             if(mode == 's'):
#                 if(tokens_only): # Test Text with or without tags
#                     yield split_sentences(remove_stopwords(line))
#                 else:
#                     yield gensim.models.doc2vec.TaggedDocument(split_sentences(remove_stopwords(line)), [i])
#             elif(mode == 'w'):  # Train text with or without tags
#                 if(tokens_only):
#                     yield gensim.utils.simple_preprocess(line)
#                 else:
#                     yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])





class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        with smart_open.smart_open(self.dirname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                # if (mode == 's'):
                #     if (tokens_only):  # Test Text with or without tags
                #         yield split_sentences(remove_stopwords(line))
                #     else:
                #         yield gensim.models.doc2vec.TaggedDocument(split_sentences(remove_stopwords(line)), [i])
                # elif (mode == 'w'):  # Train text with or without tags
                #     if (tokens_only):
                #         yield gensim.utils.simple_preprocess(line)
                #     else:
                    yield nltk.word_tokenize(f)
