import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

# %%

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.ref_dic = None

    @staticmethod
    def _get_enc(ser):
        count = ser.value_counts()
        noise = abs(np.random.normal(5e-6, 1e-5, len(count)))
        return (count / count.sum() + noise).to_dict()

    def fit(self, X, y=None):
        self.ref_dic = {}
        for col in self.cols:
            col_enc = self._get_enc(X[col])
            self.ref_dic[col] = col_enc
        return self

    def transform(self, X, y=None):
        for col in self.cols:
            X[col] = X[col].apply(
                lambda x, col=col: self.ref_dic[col].get(x, 0))
        return X

# %%

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text


class StemmedTfidfVectorizer(TfidfVectorizer):

    english_stemmer = nltk.stem.SnowballStemmer('english')

    def build_analyzer(self):
        # analyzer = super(TfidfVectorizer, self).build_analyzer()
        analyzer = TfidfVectorizer.build_analyzer(self)
        return lambda doc: (self.english_stemmer.stem(w)
                            for w in analyzer(doc))
