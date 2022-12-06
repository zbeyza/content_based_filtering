# imports and display settings

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##############################################
# 1. Generating the TF-IDF Matrix
##############################################

# uploading data
# https://www.kaggle.com/rounakbanik/the-movies-dataset

df = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)

def df_info(dataframe):
    print("             Take a Look at the Dataset            ")
    print("---------------------------------------------------")
    print(f"First 10 Observations: \n{dataframe.head(10)}")
    print("---------------------------------------------------")
    print(f"Last 10 Observations: \n{dataframe.tail(10)}")
    print("---------------------------------------------------")
    print(f"Dataframe Columns: \n{dataframe.columns}")
    print("---------------------------------------------------")
    print(f"Descriptive Statistics: \n{dataframe.describe().T}")
    print("---------------------------------------------------")
    print(f"NaN: \n{dataframe.isnull().sum()}")
    print("---------------------------------------------------")
    print(f"Variable Types: \n{dataframe.dtypes}")
    print("---------------------------------------------------")
    print(f"Number of Observations: \n{dataframe.shape[0]}")
    print(f"Number of Variables: \n{dataframe.shape[1]}")

df_info(df)

df["overview"].head()

# we want to eliminate frequently used words in english such as the, and, on
tfidf = TfidfVectorizer(stop_words="english")

# Replacing missing information in the overview variable with " "
df["overview"] = df["overview"].fillna(" ")

# eliminating frequently used words in "overview" varibale
tfidf_matrix = tfidf.fit_transform(df["overview"])

tfidf_matrix.shape
# 45466 : overviews
# 75827 : unique words
# tf-idf scores exist at the intersection of these two

# specifying the types of values that make up the matrix as float32
tfidf_matrix = tfidf_matrix.astype(np.float32)

# while creating the cosine similarity matirx, I get ArrayMemoryError, limiting the matrix will solve this error
tfidf_matrix = tfidf_matrix[:15000, :15000]

# convert matrix to array
tfidf_matrix.toarray()

##############################################
# 2.Creating the Cosine Similarity MAtrix
##############################################

# similarity values of each movie are found with the other movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape

# similarity scores of the movie in the 1st index with all the other movies
cosine_sim[1]


##############################################
# 3. Making Suggestions Based on Similarities
##############################################

# information on which movie is in which index
indices = pd.Series(df.index, index=df["title"])

indices.index.value_counts()
# It is seen that there is multiplexing in the titles.

# we need to keep one of these multiples and eliminate the rest
# we will take the most recent one of these multiples on the most recent date.
indices = indices[~indices.index.duplicated(keep="last")]

movie_index = indices["Toy Story"]
# since we limited the matrix, you should choose a movie that still in the limited matrix

# Similarity scores between Sherlock Holmes and other movies
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

# fetching the indeces of the 10 most similar movies
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
# The 0th index contains the movie itself, so we choose from the 1st index.

# fetching movie titles in these indeces
df["title"].iloc[movie_indices]