import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("test2.csv", encoding='latin1')
print(data.shape)

# using cosine_similarity algorithm
# converting the data element array into list
articles = data["Link"].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words="english")
# print(uni_tfidf)
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)
def recommend_articles(x):
    return ", ".join(data["Article"].loc[x.argsort()[-5:-1]])
data["Recommended Articles"] = [recommend_articles(x) for x in uni_sim]
# print(type(data))
# print(data[["Recommended Articles"]])
array = data["Article"].to_numpy()
name = input()
for i in range(0, len(array)):
    if name in array[i]:
        print(data.loc[[i]])
        break
# print(data.loc[[22]])
# print(data.shape)