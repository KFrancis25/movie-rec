import os
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Define file paths based on the current script location
current_dir = os.getcwd()  # Gets the current working directory
ratings_path = os.path.join(current_dir, 'rating.csv')
movies_path = os.path.join(current_dir, 'movie.csv')

# Load MovieLens dataset
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

ratings_sample = ratings.sample(n=20000, random_state=42)

# Prepare the data for the Surprise library
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define the KNN collaborative filtering algorithm
# Using item-based collaborative filtering with cosine similarity
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})

# Train the algorithm on the training set
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Calculate and print accuracy
print(f"RMSE: {accuracy.rmse(predictions):.4f}")

# Function to get movie recommendations
def get_movie_recommendations(user_id, num_recommendations=5):
    # Get a list of all unique movie IDs
    movie_ids = movies['movieId'].unique()

    # Get the list of movie IDs the user has already rated
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()

    # Filter out movies the user has already rated
    unrated_movies = [movie_id for movie_id in movie_ids if movie_id not in user_rated_movies]

    # Predict ratings for all unrated movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movies]

    # Sort predictions by estimated rating in descending order
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]

    # Get the movie titles for the recommended movie IDs
    recommended_movie_ids = [int(pred.iid) for pred in recommendations]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]

    return recommended_movies[['movieId', 'title']]

# Test the recommendation function
user_id = 1  # Example user ID
recommended_movies = get_movie_recommendations(user_id, num_recommendations=5)
print("Recommended movies for User", user_id)
print(recommended_movies)
