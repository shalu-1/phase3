# phase3
# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
# Example dataset: movies.csv
# Columns: ['MovieID', 'Title', 'Genres']
movies = pd.read_csv('movies.csv')

# Example dataset: ratings.csv
# Columns: ['UserID', 'MovieID', 'Rating']
ratings = pd.read_csv('ratings.csv')

# Preprocess Data
# Merge movies and ratings data
movie_data = pd.merge(ratings, movies, on='MovieID')

# Create a utility matrix for collaborative filtering
utility_matrix = movie_data.pivot_table(index='UserID', columns='Title', values='Rating').fillna(0)

# Content-Based Filtering
# Combine movie genres into a single string for each movie
movies['GenresCombined'] = movies['Genres'].apply(lambda x: " ".join(x.lower().split('|')))

# Use CountVectorizer to create a matrix of genres
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(movies['GenresCombined'])

# Compute the cosine similarity between movies based on genres
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to recommend movies based on content
def recommend_movies_based_on_content(movie_title, cosine_sim=cosine_sim, movies=movies):
    try:
        # Get the index of the movie that matches the title
        movie_idx = movies[movies['Title'] == movie_title].index[0]

        # Get similarity scores for all movies with the selected movie
        similarity_scores = list(enumerate(cosine_sim[movie_idx]))

        # Sort the movies based on similarity scores
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 most similar movies
        top_movies = sorted_scores[1:11]
        recommended_movies = [movies.iloc[i[0]]['Title'] for i in top_movies]

        return recommended_movies
    except IndexError:
        return f"Movie '{movie_title}' not found in the dataset."

# Collaborative Filtering
# Compute cosine similarity for users based on their ratings
user_similarity = cosine_similarity(utility_matrix)

# Function to recommend movies for a user based on collaborative filtering
def recommend_movies_for_user(user_id, utility_matrix=utility_matrix, user_similarity=user_similarity):
    try:
        # Get the user's similarity scores with all other users
        user_idx = utility_matrix.index.get_loc(user_id)
        similarity_scores = user_similarity[user_idx]

        # Get the weighted ratings
        weighted_ratings = utility_matrix.T.dot(similarity_scores) / similarity_scores.sum()

        # Recommend the top 10 movies the user hasn't rated yet
        user_rated_movies = utility_matrix.loc[user_id] > 0
        recommendations = weighted_ratings[~user_rated_movies].nlargest(10).index

        return recommendations.tolist()
    except KeyError:
        return f"User ID '{user_id}' not found in the dataset."

# Example Usage
if __name__ == "__main__":
    # Content-based recommendation example
    movie_title = "Toy Story (1995)"
    print(f"Movies similar to '{movie_title}':")
    print(recommend_movies_based_on_content(movie_title))

    # Collaborative filtering recommendation example
    user_id = 1
    print(f"Recommended movies for User {user_id}:")
    print(recommend_movies_for_user(user_id))
