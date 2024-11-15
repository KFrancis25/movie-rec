# Movie Recommendation System using Collaborative Filtering

## Overview
This project is a **movie recommendation system** that suggests movies to users based on collaborative filtering. Using the **MovieLens dataset**, we analyze user-movie ratings to identify patterns and recommend movies that similar users or similar movies have highly rated.

## Project Structure
- **Dataset**: We use the [MovieLens dataset](https://grouplens.org/datasets/movielens/), which contains user IDs, movie IDs, and ratings. This data helps the model learn user preferences.
- **Approach**: Collaborative filtering is applied in two ways:
  - **User-Based Filtering**: Recommends movies that similar users have liked.
  - **Item-Based Filtering**: Recommends movies similar to ones the user has already enjoyed.

## Tools and Libraries
- **Python**: Main programming language for data handling and modeling.
- **Pandas**: Used for data manipulation and preprocessing.
- **Surprise Library**: Specialized library for building and analyzing recommendation systems. We use `KNNBasic` for collaborative filtering.
- **Scikit-learn**: For additional data handling and evaluation support.
- **Matplotlib / Seaborn**: For optional data visualization.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/movie-recommendation-system.git
    ```
2. Install the required packages:
    ```bash
    pip install pandas scikit-surprise matplotlib
    ```

## Data Preparation
1. **Download the MovieLens dataset** and place `movies.csv` and `ratings.csv` files in the project directory.
2. **Load the Data**: Use Pandas to load and inspect the data, ensuring it's formatted as expected for the Surprise library.

## Model Training
1. **Choose Collaborative Filtering Method**: For this project, we use item-based collaborative filtering with cosine similarity.
2. **Train the Model**: Fit the model on a training set of the data using the Surprise library.
3. **Evaluate the Model**: Measure accuracy using Root Mean Square Error (RMSE) on the test set to gauge recommendation quality.

## Usage
### Running the Recommendation System
To generate movie recommendations for a specific user:
1. Set a user ID to get recommendations for that user.
2. Run the `get_movie_recommendations` function to display recommended movies based on collaborative filtering.

### Example
```python
# Example usage
user_id = 1  # Specify the user ID
recommended_movies = get_movie_recommendations(user_id, num_recommendations=5)
print("Recommended movies:")
print(recommended_movies)
```

### Project Structure
movie-recommendation-system/
├── movies.csv               # MovieLens movies data
├── ratings.csv              # MovieLens ratings data
├── movie_recommendation.py   # Main Python script for the recommendation system
└── README.md                # Project documentation

## Future Improvements

- **Hybrid Recommendation System**: Combine collaborative filtering with content-based filtering for more accurate recommendations.
- **Matrix Factorization (SVD)**: Explore matrix factorization techniques to capture latent features.
- **Parameter Tuning**: Experiment with different similarity metrics and values of `k` for better results.

## References

- **[MovieLens Dataset](https://www.kaggle.com/code/mrisdal/starter-movielens-20m-dataset-144a8ee2-e)**: The dataset used for training and evaluating the recommendation system.
- **[Surprise Library Documentation](https://surprise.readthedocs.io/en/stable/)**: Detailed documentation on collaborative filtering algorithms.

## Contributors
- Alex Chavez [@alexchavez01]
- Kai Francis [@KFrancis25]
