import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

# Charger les données
netflix_data = pd.read_csv('netflix1.csv')
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'])
netflix_data['year_added'] = netflix_data['date_added'].dt.year

# Convertir les valeurs infinies en NaN
def replace_infinities(df):
    return df.replace([float('inf'), -float('inf')], np.nan)

netflix_data = replace_infinities(netflix_data)

# Prétraitement des données
def preprocess_duration(df):
    # Remplacer ' min' par rien et convertir en numérique, gérer les erreurs
    df['duration'] = df['duration'].str.replace(' min', '', regex=False).astype(float)
    return df

netflix_movies = netflix_data[netflix_data['type'] == 'Movie'].copy()
netflix_movies = preprocess_duration(netflix_movies)

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')
netflix_movies['duration'] = imputer.fit_transform(netflix_movies[['duration']])

# Titre du tableau de bord
st.title('Tableau de Bord Netflix')

# Sidebar pour les filtres
st.sidebar.header('Filtres')
type_filter = st.sidebar.multiselect('Type', netflix_data['type'].unique(), netflix_data['type'].unique())
year_filter = st.sidebar.slider('Année de sortie', int(netflix_data['release_year'].min()), int(netflix_data['release_year'].max()), 
                                (int(netflix_data['release_year'].min()), int(netflix_data['release_year'].max())))
country_filter = st.sidebar.multiselect('Pays', netflix_data['country'].unique(), netflix_data['country'].unique())

# Application des filtres
filtered_data = netflix_data[
    (netflix_data['type'].isin(type_filter)) &
    (netflix_data['release_year'] >= year_filter[0]) &
    (netflix_data['release_year'] <= year_filter[1]) &
    (netflix_data['country'].isin(country_filter))
]

# 1. Répartition des types (Films vs Séries TV)
st.subheader('Répartition des types (Films vs Séries TV)')
fig1 = px.pie(filtered_data, names='type', title='Répartition des types (Films vs Séries TV)', color_discrete_sequence=['#E50914', '#221f1f'])
st.plotly_chart(fig1)

# 2. Nombre de contenus ajoutés par année
st.subheader('Nombre de contenus ajoutés par année')
fig2 = px.bar(filtered_data, x='year_added', title='Nombre de contenus ajoutés par année', color_discrete_sequence=['#E50914'])
st.plotly_chart(fig2)

# 3. Répartition des classements par âge
st.subheader('Répartition des classements par âge')
fig3 = px.bar(filtered_data, x='rating', title='Répartition des classements par âge', color_discrete_sequence=['#E50914'])
st.plotly_chart(fig3)

# 4. Durée des films (en minutes)
st.subheader('Durée des films (en minutes)')
fig4 = px.histogram(netflix_movies, x='duration', title='Durée des films (en minutes)', color_discrete_sequence=['#E50914'])
st.plotly_chart(fig4)

# 5. Répartition des contenus par pays
st.subheader('Répartition des contenus par pays (Top 10)')
top_countries = filtered_data['country'].value_counts().head(10).index
fig5 = px.bar(filtered_data[filtered_data['country'].isin(top_countries)], y='country', title='Répartition des contenus par pays (Top 10)', 
              color_discrete_sequence=['#E50914'], orientation='h')
st.plotly_chart(fig5)

# 6. Principaux réalisateurs en termes de nombre de contenus
st.subheader('Principaux réalisateurs en termes de nombre de contenus (Top 10)')
top_directors = filtered_data['director'].value_counts().head(10).index
fig6 = px.bar(filtered_data[filtered_data['director'].isin(top_directors)], y='director', title='Principaux réalisateurs en termes de nombre de contenus (Top 10)', 
              color_discrete_sequence=['#E50914'], orientation='h')
st.plotly_chart(fig6)

# 7. Graphique 3D des types de contenus par année et pays
st.subheader('Graphique 3D des types de contenus par année et pays')
fig7 = px.scatter_3d(filtered_data, x='year_added', y='country', z='type', color='type', title='Graphique 3D des types de contenus par année et pays')
st.plotly_chart(fig7)

# 8. Nuage de mots des titres
st.subheader('Nuage de mots des titres')
text = ' '.join(filtered_data['title'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)
fig8, ax8 = plt.subplots()
ax8.imshow(wordcloud, interpolation='bilinear')
ax8.axis('off')
st.pyplot(fig8)

# 9. Insights
st.subheader('Insights')
st.write("""
- *Répartition des types*: Il y a plus de films que de séries TV sur Netflix.
- *Contenus par année*: La plupart des contenus ont été ajoutés récemment.
- *Classements par âge*: Le classement TV-MA est le plus courant.
- *Durée des films*: La durée moyenne des films est de 90-100 minutes.
- *Pays d'origine*: Les États-Unis dominent la production de contenus.
- *Réalisateurs*: Les réalisateurs les plus prolifiques ont dirigé de nombreux contenus.
""")