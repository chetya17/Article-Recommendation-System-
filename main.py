import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("C:\\Users\CHETAN\\Desktop\\google sdg challenge.csv", encoding='latin1')
articles = data["Article"].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)
def recommend_articles(x):
    return ", ".join(data["Title"].loc[x.argsort()[-5:-1]])
data["Recommended Articles"] = [recommend_articles(x) for x in uni_sim]
array = data["Title"].to_numpy()
name = input()
for i in range(0, len(array)):
    if name in array[i]:
        print(data.loc[[i]])
        break