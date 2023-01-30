import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from streamlit_functions import *

### Define page-wide parameters
BGCOLOR = 'lightskyblue'
FONTSIZE = 17
FONTFAMILY = 'Garamond'
WIDTH_TO_CENTER = 3
PERSIST = False
CACHESUPPRESS = True

st.sidebar.markdown(":ribbon: Build your own Film :ribbon:")


### Let user select film by genome tags
### Start the Page
_, col2, _ = st.columns([0.1, 0.65, 0.1])
with col2:
    st.title(":ribbon: Find a Film by Tags :ribbon:")

_, col2, _ = st.columns([0.1, 0.25, 0.1])
with col2:
    st.video("https://www.youtube.com/watch?v=JlQh4PeB8PE")

st.write("This page is based on the tag genome data contained in the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/). \
There are a total of 1128 tags, each with a relevance score labeling ~12.000 films. According to the authors:\
    \"The tag genome was computed using a machine learning algorithm on user-contributed content including tags, ratings, and textual reviews.\" \
    Detailed information on the implementation can be found inside their [publication](https://dl.acm.org/doi/10.1145/2362394.2362395).")


@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_genome_data():
    return load_genome_data()

# Load film data
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_movie_filtered_director():
    return load_movie_filtered_director()
#movies = st_load_movie_filtered_director()

@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_pure_tags_split(k):
    return load_pure_tags_split(k)

#genome = st_load_genome_data()  # this is filtered to relevance >= 0.5
genome = load_movie_filtered_director()  # this is filtered to relevance >= 0.5
#pure_tags = st_load_pure_tags_split()  # this is unfiltered

tag_list_total = genome['tag'].unique()
tag_list_user = st.multiselect(label='Choose which tags you want to be contained in the film:',
						options=tag_list_total,
						default="great ending",
						max_selections=10
)

# calculate total relevance combining all user-input tags
user_subset_tags = pd.DataFrame()
for k in range(5):
    pure_tags = st_load_pure_tags_split(k)
    tmp = pure_tags[pure_tags['tag'].isin(tag_list_user)]
    user_subset_tags = pd.concat([user_subset_tags, tmp], axis=0)

user_subset_tags = user_subset_tags[['relevance','imdbId']].groupby('imdbId').mean().sort_values(by='relevance', ascending=False) #imdb id is index now
user_subset_tags = user_subset_tags.rename(columns={'relevance':'meanRelevance'})
# pick top 50 suggestions
user_subset_tags = user_subset_tags.iloc[0:50]
# find matching movies in extended dataset
user_tag_recommendations = genome[genome['imdbId'].isin(user_subset_tags.index)]
# select the user chosen tags in extended dataset
user_tag_recommendations = user_tag_recommendations[user_tag_recommendations['tag'].isin(tag_list_user)]
# ignore tag repetition and get unqiue movies
user_tag_recommendations = user_tag_recommendations.drop_duplicates(subset=['imdbId'])
# join the proper mean relevance
user_tag_recommendations = user_tag_recommendations.join(user_subset_tags, on='imdbId')
# extract important features
user_tag_recommendations = user_tag_recommendations[['imdbId','meanRelevance','primaryTitle','averageRating','numVotes','startYear','runtimeMinutes','genres','directorsName','tagline']]
user_tag_recommendations = user_tag_recommendations.set_index('imdbId', drop=True)
user_tag_recommendations = user_tag_recommendations.rename(columns={'meanRelevance':'Mean Relevance', 'primaryTitle':'Title', 'startYear':'Year', 'runtimeMinutes':'Runtime [Min]', 'genres':'Genres', 'averageRating':'IMDB Rating', 'numVotes':'IMDB Votes', 'directorsName':'Director', 'tagline':'Tagline'})
user_tag_recommendations = user_tag_recommendations.sort_values(by='Mean Relevance', ascending=False)

st.subheader('Top 50 results matching your tags by mean relevance:')
st.dataframe(user_tag_recommendations)