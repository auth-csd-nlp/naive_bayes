import math

import nltk

nltk.download('punkt')

docs = []
docs.append(['just plain boring', 'entirely predictable and lacks energy', 'no surprises and very few laugs'])
docs.append(['very powerful', 'the most fun film of the summer'])
test_doc = 'the film was predictable with no fun'


class NB:

    def __init__(self, binary=False, bernoulli=False):
        self.priors = []
        self.probs = []
        self.vocabulary = []
        self.lengths = []
        self.num_classes = 0
        self.binary = binary
        self.bernoulli = bernoulli

    def fit(self, docs):
        self.num_classes = len(docs)
        all_docs = 0
        for class_docs in docs:
            all_docs += len(class_docs)
        for class_docs in docs:
            self.priors.append(len(class_docs) / all_docs)
        counts = []
        for class_docs in docs:
            count = dict()
            s = 0
            for doc in class_docs:
                tokens = nltk.word_tokenize(doc)
                s = s + len(tokens)
                for token in tokens:
                    if token in count:
                        count[token] += 1
                    else:
                        count[token] = 1
                    if token not in self.vocabulary:
                        self.vocabulary.append(token)
            self.lengths.append(s)
            counts.append(count)
        for i in (0, self.num_classes - 1):
            prob = dict()
            for token in counts[i]:
                prob[token] = (counts[i][token] + 1) / (self.lengths[i] + len(self.vocabulary))
            self.probs.append(prob)

    def predict_proba(self, doc):
        tokens = nltk.word_tokenize(doc)
        scores = []
        sum = 0
        for i in (0, self.num_classes - 1):
            score = 0
            score += math.log(self.priors[i])
            for token in tokens:
                if token in self.probs[i]:
                    score += math.log(self.probs[i].get(token))
                else:
                    if token in self.vocabulary:
                        score += math.log(1 / (self.lengths[i] + len(self.vocabulary)))
            sum += math.exp(score)
            scores.append(math.exp(score))

        for i in (0, self.num_classes - 1):
            scores[i] = scores[i] / sum

        return scores


def test(binary=False, bernoulli=False):
    """
    >>> test()
    [0.1841513848957118, 0.8158486151042883]
    >>> test(binary=True)
    [0.221238739047567, 0.7787612609524331]
    >>> test(bernoulli=True)
    [0.12153600386063369, 0.8784639961393663]
    """
    nb = NB(binary=binary, bernoulli=bernoulli)
    nb.fit(docs)
    scores = nb.predict_proba('the film was predictable with no fun')
    print(scores)
