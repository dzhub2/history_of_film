# Imports
import streamlit as st
from  streamlit_functions import *

import matplotlib.pyplot as plt

### Define page-wide parameters
BGCOLOR = 'lightskyblue'
FONTSIZE = 15
FONTFAMILY = 'Garamond'
WIDTH_TO_CENTER = 3
PERSIST = False
CACHESUPPRESS = True

st.sidebar.markdown(":film_projector: Welcome Page :film_projector:")

# ### Load film data
# @st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
# def st_load_movie_data_from_genome():
#     return load_movie_data_from_genome()
# movies = st_load_movie_data_from_genome()

### Load film data
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_movie_filtered_director():
    return load_movie_filtered_director()
movies = st_load_movie_filtered_director()

### Start the Page
_, col2, _ = st.columns([0.215, 1.5, 0.1])
with col2:
    st.title(":film_projector: The History of Film :film_projector:")

_, col2, _ = st.columns([0.1, 0.25, 0.1])
with col2:
    st.video("https://www.youtube.com/watch?v=BXsWn9DhF5g")

### Display a random movie suggestion
col1, col2 = st.columns([3,5])
with col2:
    if st.button("Give me a Film!"):
        pass

mid1, col1, mid2, col2 = st.columns([2,1,10,10])
mymovie = movies.sample(1)
with col1:
    imageLocation = st.empty()
    imageLocation.image("https://image.tmdb.org/t/p/original" + str(mymovie['poster_path'].iloc[0]), width=230)
with col2:
    st.write(f":movie_camera: **Title** : {mymovie['primaryTitle'].iloc[0]}")
    st.write(f":100: **IMDB Rating** : {mymovie['averageRating'].iloc[0]}")
    st.write(f":vampire: IMDB Votes : {mymovie['numVotes'].iloc[0]}")
    st.write(f":hourglass_flowing_sand: **Year** : {mymovie['startYear'].iloc[0]}")
    st.write(f":stopwatch: **Runtime** : {mymovie['runtimeMinutes'].iloc[0]}")
    st.write(f":dragon: **Genres** : {mymovie['genres'].iloc[0]}")
    st.write(f":film_frames: **Director** : {mymovie['directorsName'].iloc[0]}")
    st.write(f":stars: **Tagline** : {mymovie['tagline'].iloc[0]}")

##################################################################################
### Let the user search the DB for a movie
user_movie = st.selectbox(label='Enter a film title that you\'re curious about',
						options=movies['primaryTitle'].sort_values(),
						index=31
)

mid1, col1, mid2, col2 = st.columns([2,1,10,10])
mymovie = movies[movies['primaryTitle']==user_movie]

if len(mymovie) == 0:
    st.write("The movie you entered isn't in the database :smiling_face_with_tear:")
else:
    with col1:
        imageLocation = st.empty()
        imageLocation.image("https://image.tmdb.org/t/p/original"+mymovie['poster_path'].iloc[0], width=230)
    with col2:
        st.write(f":movie_camera: **Title** : {mymovie['primaryTitle'].iloc[0]}")
        st.write(f":100: **IMDB Rating** : {mymovie['averageRating'].iloc[0]}")
        st.write(f":vampire: **IMDB Votes** : {mymovie['numVotes'].iloc[0]}")
        st.write(f":hourglass_flowing_sand: **Year** : {mymovie['startYear'].iloc[0]}")
        st.write(f":stopwatch: **Runtime** : {mymovie['runtimeMinutes'].iloc[0]}")
        st.write(f":dragon: **Genres** : {mymovie['genres'].iloc[0]}")
        st.write(f":film_frames: **Director** : {mymovie['directorsName'].iloc[0]}")
        st.write(f":stars: **Tagline** : {mymovie['tagline'].iloc[0]}")

### Create a word cloud of all movie titles
_, col2, _ = st.columns([0.4, 1.5, 0.2])
with col2:
    st.subheader("Most Common Words in Film Titles")

@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_create_wordcloud(movies):
    return create_wordcloud(movies)
title_wordcloud = st_create_wordcloud(movies)


fig, ax = plt.subplots()
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()
st.pyplot(fig)
