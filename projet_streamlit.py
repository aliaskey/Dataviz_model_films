import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('https://raw.githubusercontent.com/VictoriaGaullier/Reco-films/main/data.csv')
st.title('Notre système de recommandation de films')
films = st.selectbox('Quel film avez-vous aimé ?', options=df['title'])
# Concatenate columns into a single string
df['concatenated'] = (4*df['rated']).astype(str) + ' ' + df['genre_1'] + ' ' + df['actor_1'] + ' ' + df['plot'] + ' ' + df['genre_2']
# Fit TfidfVectorizer on concatenated string
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), stop_words=['and', 'the', 'a', 'an', 'in', 'of', 'to', 'is', 'it', 'that'])
tfv_matrix = tfv.fit_transform(df['concatenated'])
# Compute sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Create mapping of movie titles to their index in the dataframe
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
# Function to give recommendations based on input movie
def give_rec(title, sig=sig):
    # get index corresponding to the original_title
    idx = indices[title]
    # Get the list of ids along with pairwise similarity scores of the provided idx with other ids
    # Sort the movies
    # Selecting top 5 movies for recommendation
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:6]
    # Movie indices
    movies_indices = [i[0] for i in sig_scores]
    # Top 5 similar movies
    return df['title'].iloc[movies_indices]
# Get recommendations for the selected movie
rec_movies = give_rec(films)
st.write('Films recommandés :')
for idx, movie in rec_movies.items():
    st.write(movie)
    poster_url = df.loc[df['title'] == movie, 'poster'].iloc[0]
    st.image(poster_url)
    