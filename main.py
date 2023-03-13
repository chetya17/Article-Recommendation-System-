import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("Dataset upload karoo", encoding='latin1')
data.head()