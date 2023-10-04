# code 
import numpy as np 
import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt 
import seaborn as sns 

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 

ratings = pd.read_csv("ratings.csv") 
ratings.head() 

movies = pd.read_csv("movies.csv") 
movies.head() 

noOf_ratings = len(ratings) 
noOf_movies = len(ratings['movieId'].unique()) 
noOf_users = len(ratings['userId'].unique()) 

print(f"Number of ratings: {noOf_ratings}") 
print(f"Number of unique movieId's: {noOf_movies}") 
print(f"Number of unique users: {noOf_users}") 
print(f"Average ratings per user: {round(noOf_ratings/noOf_users, 2)}") 
print(f"Average ratings per movie: {round(noOf_ratings/noOf_movies, 2)}") 

userFrequency = ratings[['userId', 'movieId']].groupby('userId').count().reset_index() 
userFrequency.columns = ['userId', 'noOf_ratings'] 
userFrequency.head() 



mean_rating = ratings.groupby('movieId')[['rating']].mean() 

lowest_ratedMovie = mean_rating['rating'].idxmin() 
movies.loc[movies['movieId'] == lowest_ratedMovie] 

highest_ratedMovie = mean_rating['rating'].idxmax() 
movies.loc[movies['movieId'] == highest_ratedMovie] 

ratings[ratings['movieId']==highest_ratedMovie] 

ratings[ratings['movieId']==lowest_ratedMovie] 


movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean']) 
movie_stats.columns = movie_stats.columns.droplevel() 


from scipy.sparse import csr_matrix 

def create_matrix(df): 
	
	N = len(df['userId'].unique()) 
	M = len(df['movieId'].unique()) 
	
	
	user_mapper = dict(zip(np.unique(df["userId"]), list(range(N)))) 
	movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M)))) 
	
	
	user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"]))) 
	movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"]))) 
	
	user_index = [user_mapper[i] for i in df['userId']] 
	movie_index = [movie_mapper[i] for i in df['movieId']] 

	X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N)) 
	
	return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper 

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings) 

from sklearn.neighbors import NearestNeighbors 
""" 
Find similar movies using KNN 
"""
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False): 
	
	neighbour_ids = [] 
	
	movie_ind = movie_mapper[movie_id] 
	movie_vec = X[movie_ind] 
	k+=1
	kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric) 
	kNN.fit(X) 
	movie_vec = movie_vec.reshape(1,-1) 
	neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance) 
	for i in range(0,k): 
		n = neighbour.item(i) 
		neighbour_ids.append(movie_inv_mapper[n]) 
	neighbour_ids.pop(0) 
	return neighbour_ids 


movie_titles = dict(zip(movies['movieId'], movies['title'])) 

movie_id = int(input("You have watched: "))

similar_ids = find_similar_movies(movie_id, X, k=10) 
movie_title = movie_titles[movie_id] 

print(f"Since you watched {movie_title}") 
for i in similar_ids: 
	print("Then you might also like: ")
	print(movie_titles[i]) 
