# Content-Based Filtering

Content-Based Filtering is one of the methods used as a Recommendation System. Similarities are calculated over product metadata, and it provides the opportunity to develop recommendations. The products that are most similar to the relevant product are recommended.

Metadata represents the features of a product/service. For example, the director, cast, and screenwriter of a movie; the author, back cover article, translator of a book, or category information of a product.

#### Steps of Content-Based Filtering:
##### 1. Represent Texts Mathematically (Text Vectorization):

- Count Vector 
- TF-IDF

##### 2. Calculate Similarities

- There is more than one way to calculate similarity. I preferred the cosine similarity method for this project.

### Problem:
A newly established online movie platform wants to make movie recommendations to its users. Since the login rate of users is very low, the habits of the users are unknown. however, the information about which movies the users watch can be accessed from the traces in the browser. According to this information, it is desired to make movie recommendations to the users.

### About Dataset:
The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

##### https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

### You can find the detailed explanation of Content-Based Filtering  and the project from the link link:
##### https://medium.com/@zbeyza/recommendation-systems-content-based-filtering-e19e3b0a309e

