from sklearn.feature_extraction.text import CountVectorizer
train = ["John likes to watch movies. Mary likes movies too"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train)

print(vectorizer.vocabulary_)
print(X.toarray())