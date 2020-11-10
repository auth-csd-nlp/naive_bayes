from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

docs = []
docs.append(['just plain boring', 'entirely predictable and lacks energy', 'no surprises and very few laugs'])
docs.append(['very powerful', 'the most fun film of the summer'])
test_doc = 'the film was predictable with no fun'

X = []
y = []
x = ['the film was predictable with no fun']

for i in (0, len(docs)-1):
    for doc in docs[i]:
        X.append(doc)
        y.append(i)

vec = CountVectorizer(binary=False)
vec.fit(X)
mnb = MultinomialNB()
mnb.fit(vec.transform(X), y)
scores = mnb.predict_proba(vec.transform(x))
print("Multinomial: ", scores)

vec = CountVectorizer(binary=True)
vec.fit(X)
bmnb = MultinomialNB()
bmnb.fit(vec.transform(X), y)
scores = bmnb.predict_proba(vec.transform(x))
print("Binary Multinomial: ", scores)

vec = CountVectorizer(binary=False)
vec.fit(X)
bnb = BernoulliNB()
bnb.fit(vec.transform(X), y)
scores = bnb.predict_proba(vec.transform(x))
print("Bernoulli: ", scores)