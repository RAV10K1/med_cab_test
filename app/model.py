import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English


# NLP Based Recommendation Bot - Courtesy Curdt Million and Ravi Tennekone

KNN = "KNN_Model.pkl"
CNN = "CNN_Model.pkl"
data = pd.read_csv("app\df_strains.csv")
path = "app/KNN_Model.pkl"

with open(path, 'rb') as nn_pickled_model:
    model_1 = pickle.load(nn_pickled_model)


class BudTender:
    """NLP based bot for recommendation of cannabis strains
    based on user input using K-Nearest Neighbors.

    Parameters :

    User input as dtype string

    Returns :

    Dict: Most similar cannabis strain(s) with description,
    name, type, effects and flavors

    """

    def __init__(self):
        self.KNN = loaded_knn
        self.data = data

        class Tokens:
            nlp = English()

            @classmethod
            def tokenize(cls, document):
                doc = cls.nlp(document)
                return [token.lemma.strip() for token in doc
                        if not token.is_stop and not token.is_punct
                        ]

        self.tfidf = TfidfVectorizer(tokenizer=Tokens.tokenize,
                                     stop_words="english",
                                     min_df=0.025, max_df=0.98,
                                     ngram_range=(1, 3))

        def recommend(self, user_input: str):
            vects = self.tfidf.transform([user_input]).todense()
            prediction = self.KNN.kneighbors(vects, return_distance=False)[1][0][0]
            output = self.data.iloc[prediction]
            recommendation = output.drop(
                ['name', 'ailment', 'all_text', 'lemmas', 'Cos Sim Strains', 'KNN Strains']).to_dict()

            return recommendation


if __name__ == '__main__':
    bot = BudTender()
