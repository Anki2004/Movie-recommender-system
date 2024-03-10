import pandas as pd
import streamlit as st
import pickle as pk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
movies_dic = pk.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dic)
similarity = cosine_similarity
tfidf_metrics = pk.load(open('tfidf_metrics.pkl','rb'))
vectorizer = pk.load(open('vectorizer.pkl','rb'))
def recommend_movies(input_text, user_rating, num_recommendations=5):
    input_vector = vectorizer.transform([input_text])
    similarities = similarity(input_vector, tfidf_metrics)

    # Using argsort to get indices of movies sorted by similarity
    sorted_indices = similarities.argsort()[0][-num_recommendations:][::-1]

    recommended_movies = []
    recommended_movies_rating = []
    for index in sorted_indices:
        recommended_movies.append(movies.iloc[index].title)
        recommended_movies_rating.append(movies.iloc[index].rating)
        movie_id = movies.iloc[index].movieId
        

    return recommended_movies, recommended_movies_rating


st.title("Movie Recommendation System")
selected_movie_name = st.selectbox(
    'Search your genre here',
    movies['genres'].values)
movie_rating = st.number_input('Any desired rating?')

if st.button('Recommend'):
    recommendation, rating = recommend_movies(selected_movie_name, movie_rating)
    for i,r in zip(recommendation, rating):
        st.write(f"{i} - Rating: {r}")