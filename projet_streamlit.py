import googletrans
from googletrans import Translator
import plotly.express as px
from plotly.figure_factory import create_distplot
import plotly.figure_factory as ff
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.express as px
import numpy as np
import altair as alt
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import warnings
import webbrowser
import toml
df = pd.read_csv('https://raw.githubusercontent.com/aliaskey/projet_Reco_films/main/data3.csv?token=GHSAT0AAAAAACCXIQFBVODZEVSKCKCKFI4WZDM4AKA')

##################################################################################################################
##################################################################################################################
tab = ["Choisir un onglet","User","Dashboarding", "BDD"]
selected_tab = st.sidebar.selectbox("Menu", tab, index=0)
###################################################################################################################
# Afficher une image
if selected_tab == "Choisir un onglet":
    film_image = "https://drive.google.com/uc?id=1I8680714dVA9m-LgUZf8Lwe_1QxCvlJL"
    st.image(film_image, width=400, use_column_width=True)
    st.write("# Comment utiliser l'application")
    # Création de conteneurs pour la présentation
    col1, col2 = st.columns(2)
    with col1:
        st.header('Quelques statistiques sur ...')
        st.subheader("les films (type, durée), acteurs (nombre de films, type de films) et d’autres.")
    with col2:
        st.header('Une liste de films recommandés en fonction ...')
        st.subheader(" d'IDs ou de noms de films choisis par un utilisateur.")
###########################################################################################################################
if selected_tab == "User":
        warnings.filterwarnings('ignore')
        # df = pd.read_csv('https://raw.githubusercontent.com/VictoriaGaullier/Reco-films/main/data.csv')
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
      # ###########################################################################################################################
# Contenu de onglet 3 pour afficher la page 3
elif selected_tab == "Dashboarding":
    #image de presentation
    film_image_2 = "https://drive.google.com/uc?id=1oVoHH7Q78r8h4Iunnym2mSkKJjcV-NSl"
    st.image(film_image_2, width=400, use_column_width=True)
    st.header("Dashboarding")
    with st.expander("CinéStats"): 
        # st.title("Statistiques sur les films")
        st.markdown(f"<h1 style='text-align: center;'>Statistiques sur les films</h1>",unsafe_allow_html=True,)
        # graphiques 
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition des notes moyennes des films")
            chart_data = pd.DataFrame({"averageRating": df["averageRating"], "numVotes": df["numVotes"]})
            chart = alt.Chart(chart_data).mark_circle().encode(
                x=alt.X('averageRating', scale=alt.Scale(domain=[0, 10])),
                y='numVotes', tooltip=['averageRating', 'numVotes'])
            st.altair_chart(chart, use_container_width=True)
        with col2:
            st.subheader("GDP vs Life expectancy (2007)")
            df = px.data.gapminder()
            fig = px.scatter(
                df.query("year==2007"),
                x="gdpPercap",
                y="lifeExp",
                size="pop",
                color="continent",
                hover_name="country",
                log_x=True,
                size_max=60,)
            tab1, tab2 = st.columns(2)
            with tab1:
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            with tab2:
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        # Second column
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition des notes moyennes des films")
            chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c'])
            st.area_chart(chart_data)

#         with col2:
            # st.subheader("Nombre de films par année")
            # df_yearly = df[df['startYear'] < 2023]['startYear'].value_counts().sort_index().reset_index()
            # df_yearly.columns = ['startYear', 'count']
            # st.line_chart(df_yearly.set_index('startYear'))

        # Third column
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Graphique en barre")
            chart_data = pd.DataFrame(np.random.randn(20), columns=["averageRating"])
            st.bar_chart(chart_data["averageRating"])

        with col2:
            st.subheader("Histogramme")
            # Add histogram data
            x1 = np.random.randn(200) - 2
            x2 = np.random.randn(200)
            x3 = np.random.randn(200) + 2
            # Group data together
            hist_data = [x1, x2, x3]
            group_labels = ['Group 1', 'Group 2', 'Group 3']
            # Create distplot with custom bin_size
            fig = ff.create_distplot(
                hist_data, group_labels, bin_size=[.1, .25, .5])
            # Plot!
            st.plotly_chart(fig, use_container_width=True)

############################################################################################################################################
 # Top 10 des films les mieux notés chez votre concurrent
    with st.expander("Les meilleurs films sur IMDb"):
            # Créer un objet Translator
            translator = Translator()
            # URL de la page IMDb à parcourir
            url_IMDb = "https://www.imdb.com/chart/top/?ref_=nv_mv_250"
            # Envoyer une requête GET à l'URL et récupérer le contenu de la réponse
            response = requests.get(url_IMDb)
            # Analyser le contenu HTML avec BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Initialiser les listes pour stocker les titres et les notes des films
            titles_IMDb = []
            ratings_IMDb = []
            # Sélectionner tous les éléments <a> dans les éléments <td> avec la classe "titleColumn"
            movie_titles = soup.select('td.titleColumn a')
            # Parcourir les 10 premiers éléments <a> et traduire les titres en français
            for title in movie_titles[:10]:
                title_fr = translator.translate(title.text, dest='fr').text
                titles_IMDb.append(title_fr)
            # Sélectionner tous les éléments <strong> dans les éléments <td> avec la classe "ratingColumn"
            movie_ratings = soup.select('td.ratingColumn strong')
            # Parcourir les 10 premiers éléments <strong> et récupérer les notes des films
            for rating in movie_ratings[:10]:
                ratings_IMDb.append(rating.text)
            # Créer un dictionnaire contenant les données pour les 10 premiers films
            data_IMDb = {'Top10': range(1, 11), 'Titre_IMDb': titles_IMDb, 'Note_IMDb': ratings_IMDb}
            # Créer un DataFrame à partir du dictionnaire
            imdb_df = pd.DataFrame(data_IMDb)
            # Afficher le titre de l'application
            st.title("Top 10 des films les mieux notés sur IMDb")
            # st.markdown(f"<h1 style='text-align: center;'>Top 10 des films les mieux notés sur IMDb</h1>",unsafe_allow_html=True,)
            # Afficher les données avec les titres traduits en français et masquer les index
            st.write(imdb_df.set_index('Top10').rename(columns={'Titre_IMDb': 'Titre', 'Note_IMDb': 'Note'}))

#############################################################################################################################################
    #visualisation de la base de données
elif selected_tab == "BDD":   
           # Afficher le titre de la section
            # st.title("Visualisation des données de l'application :")
    st.markdown(f"<h1 style='text-align: center;'>Base de données IMDB</h1>",unsafe_allow_html=True,)
            # Afficher le nombre de lignes et de colonnes dans le DataFrame
    st.write(f"Nombre de lignes : {df.shape[0]:_d} lignes")
    st.write(f"Nombre de colonnes : {df.shape[1]} colonnes")
    st.write(df)

    # Charger les valeurs du fichier config.toml dans un dictionnaire
    config = toml.load("config.toml")
    # Définir les couleurs de l'application en utilisant les valeurs du fichier config.toml
    st.set_page_config(
        primary_color=config["colors"]["primary_color"],
        background_color=config["colors"]["background_color"],
        secondary_background_color=config["colors"]["secondary_background_color"],
        text_color=config["colors"]["text_color"],
        font=config["colors"]["font"]
        )
