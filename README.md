
# Movie Recommendation System

This Movie Recommendation System suggests movies based on user input for genre and desired rating. It uses a TF-IDF (Term Frequency-Inverse Document Frequency) approach to vectorize movie descriptions and computes cosine similarity to recommend similar movies.

## Overview

The Movie Recommendation System uses collaborative filtering and content-based filtering techniques to generate movie recommendations. It takes into account user ratings, movie genres, and similarity measures to deliver relevant suggestions.
## Files
- `movies_dict.pkl`: Pickled file containing a dictionary of movie data.
- `cosine_similarity.pkl`: Pickled file containing cosine similarity function or object.
- `tfidf_metrics.pkl`: Pickled file containing TF-IDF metrics (make sure it's callable if it's a function).
- `vectorizer.pkl`: Pickled file containing the TF-IDF vectorizer object.
## Dependencies

- pandas
- streamlit
- scikit-learn
