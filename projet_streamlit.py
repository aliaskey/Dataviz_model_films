import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/VictoriaGaullier/Reco-films/main/data.csv')

st.title('Notre système de recommandation de films')

selected_films = st.multiselect('Quels films avez-vous aimés ?', options=df['title'].tolist(), key='movie_input', format_func=lambda x: x)

if selected_films:
    films = selected_films[0]
else:
    films = None


# Concaténer les colonnes en une seule chaîne de caractères
df['concatenated'] = (4 * df['rated']).astype(str) + ' ' + df['genre_1'] + ' ' + df['actor_1'] + ' ' + df['plot'] + ' ' + df['genre_2']

# Ajuster le TfidfVectorizer sur la chaîne concaténée
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words=['and', 'the', 'a', 'an', 'in', 'of', 'to', 'is', 'it', 'that'])
tfv_matrix = tfv.fit_transform(df['concatenated'])

# Calcul du noyau sigmoïde
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Créer une correspondance entre les titres des films et leur index dans la base de données
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Fonction pour donner des recommandations basées sur le film d'entrée
def give_rec(title, sig=sig) :
    # Obtenir l'index correspondant au titre original
    idx = indices[title]
    
    # Obtenir la liste des ids avec les scores de similarité par paire de l'idx fourni avec d'autres ids
    # Trier les films
    # Sélection des 5 meilleurs films pour la recommandation
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x : x[1], reverse=True)
    sig_scores = sig_scores[1:6]
    
    # Indices des films
    movies_indices = [i[0] for i in sig_scores]
    
    # Les 5 films les plus similaires
    return df['title'].iloc[movies_indices]

# Obtenir des recommandations pour le film sélectionné si un film est sélectionné
if films :
    rec_movies = give_rec(films)
else :
    rec_movies = pd.Series()

st.write('Films recommandés :')

# Créer une mise en page avec des images sur une ligne horizontale
columns = st.columns(5)

for idx, movie in rec_movies.iteritems():
    with columns[idx % 5]:
        st.write(movie)
        poster_url = df.loc[df['title'] == movie, 'poster'].iloc[0]
        st.image(poster_url, width=150)

    
