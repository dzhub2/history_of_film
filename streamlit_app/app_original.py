import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from  plotting_functions import *

### Define functions and global parameters
BGCOLOR = 'lightskyblue'
FONTSIZE = 15
FONTFAMILY = 'Garamond'
WIDTH_TO_CENTER = 3
PERSIST = False
CACHESUPPRESS = True

def write_text_center(text, WIDTH_TO_CENTER=WIDTH_TO_CENTER):
    col1, col2 = st.columns([WIDTH_TO_CENTER,5])
    with col2:
        st.write(text)
    return True

### Load and wrangle data for first plot (highest rated movies - year)
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_movie_data_from_genome():
    return load_movie_data_from_genome()
movies = st_load_movie_data_from_genome()

### Start the Dashboard
col1, col2 = st.columns([2,5])
with col2:
    st.title("The History of Film")

### Display a random movie suggestion
col1, col2 = st.columns([3,5])
with col2:
    if st.button("Give me Movie!"):
        pass

mid1, col1, mid2, col2 = st.columns([2,1,10,10])
mymovie = movies.sample(1)
with col1:
    imageLocation = st.empty()
    imageLocation.image("https://image.tmdb.org/t/p/original"+mymovie['poster_path'].iloc[0], width=230)
with col2:
    st.write(f":movie_camera: Title : {mymovie['primaryTitle'].iloc[0]}")
    st.write(f":hourglass_flowing_sand: Year : {mymovie['startYear'].iloc[0]}")
    st.write(f":stopwatch: Runtime : {mymovie['runtimeMinutes'].iloc[0]}")
    st.write(f":dragon: Genres : {mymovie['genres'].iloc[0]}")
    st.write(f":100: IMDB Rating : {mymovie['averageRating'].iloc[0]}")
    st.write(f":dragon: IMDB Votes : {mymovie['numVotes'].iloc[0]}")
    st.write(f":dragon: IMDB Votes : {mymovie['directors'].iloc[0]}")
    st.write(f":dragon: IMDB Votes : {mymovie['tagline'].iloc[0]}")

##################################################################################
### Let the user search the DB for a movie
user_movie = st.selectbox(label='Enter a film title that you\'re curious about',
						options=movies['primaryTitle'].sort_values(),
						index=29
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
        st.write(f":movie_camera: Title : {mymovie['primaryTitle'].iloc[0]}")
        st.write(f":hourglass_flowing_sand: Year : {mymovie['startYear'].iloc[0]}")
        st.write(f":stopwatch: Runtime : {mymovie['runtimeMinutes'].iloc[0]}")
        st.write(f":dragon: Genres : {mymovie['genres'].iloc[0]}")
        st.write(f":100: IMDB Rating : {mymovie['averageRating'].iloc[0]}")
        st.write(f":dragon: IMDB Votes : {mymovie['numVotes'].iloc[0]}")
        st.write(f":dragon: IMDB Votes : {mymovie['directors'].iloc[0]}")
        st.write(f":dragon: IMDB Votes : {mymovie['tagline'].iloc[0]}")

##################################################################################
### Plot: Lineplot: Highest rated movie and num votes by year

@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_highest_rated_movies_lineplot(movies):
    return wrangle_highest_rated_movies_lineplot(movies)
max_rating_year, nr_movies_per_year, total_ratings_per_year, \
norm_votes_per_year, avg_rating_per_year, max_votes_year = st_wrangle_highest_rated_movies_lineplot(movies)

fig_highest_rated = go.Figure()
# best movie per year
fig_highest_rated.add_trace(go.Scatter(x=max_rating_year.startYear, y=max_rating_year.averageRating,
					name = 'Best movies', yaxis='y', mode="markers+lines", customdata=np.stack(max_rating_year.primaryTitle, axis=-1),
					hovertemplate="<br>".join([
                        "Title: %{customdata}",
                        "Year: %{x}",
                        "IMDB Rating: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# mean rating per year
fig_highest_rated.add_trace(go.Scatter(x=avg_rating_per_year.index, y=avg_rating_per_year.meanAverageRating,
					name = 'Mean ratings', mode="markers+lines",
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean rating: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# mean votes per year
fig_highest_rated.add_trace(go.Scatter(x=norm_votes_per_year.index, y=norm_votes_per_year.averageNrRatings,
					name = 'Mean votes', yaxis="y2", mode="markers+lines",
					hovertemplate="<br>".join([
                        "Mean votes: %{y}",
                        "Year: %{x}"
						"<extra></extra>"
                    ])))
# max votes per year
fig_highest_rated.add_trace(go.Scatter(x=max_votes_year.startYear, y=max_votes_year.numVotes/10,
					name = 'Max. votes/10', yaxis="y2", mode="markers+lines", customdata=np.stack(max_votes_year.primaryTitle, axis=-1),
					hovertemplate="<br>".join([
                        "Max. votes/10: %{y}",
                        "Year: %{x}",
						"Title: %{customdata}"
						"<extra></extra>"
                    ])))

# Create axis objects
fig_highest_rated.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="IMDB Rating",
		titlefont=dict(color="#1f77b4"),
		tickfont=dict(color="#1f77b4"),
        showgrid=False),
				
	#create 2nd y axis	
	yaxis2=dict(title="Votes",overlaying="y",
				side="right",
				titlefont=dict(color="red"),
				tickfont=dict(color="red"),
                showgrid=False),
	xaxis=dict(title="Year")
)

fig_highest_rated.update_layout(
	title_text="Ratings and Votes by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.27, xanchor="right", x=0.9, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
##################################################################################
### Plot: Bar: The number of movies (in percent) per year as an animation by genre
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_count_genre_year_bar(movies, nr_movies_per_year):
    return wrangle_count_genre_year_bar(movies, nr_movies_per_year)
genres_bar_percent, genre_list = st_wrangle_count_genre_year_bar(movies, nr_movies_per_year)

fig_genre_freq = px.bar(genres_bar_percent, x='startYear', y='count',
    animation_frame=genres_bar_percent.genre, range_y=[0,80], title='Percentage of Genre per Year')
fig_genre_freq.update_layout(
    font_family=FONTFAMILY,
    font_size=FONTSIZE-2,
    plot_bgcolor = 'rgba(0, 0, 0, 0)'
)
fig_genre_freq.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000 # animation transition speed

myx = movies['startYear'].sort_values().unique()[0::2] # only plot every 2nd year
#fig_genre_freq.update_traces(textfont_size=FONTSIZE, textfont_family= FONTFAMILY, textangle=0, textposition="outside", cliponaxis=False)
fig_genre_freq.update_xaxes(tickmode='array', tickangle=75, title='Year', tickvals=myx)
fig_genre_freq.update_yaxes(tickmode='auto', tickangle=0, title='Percent')

sliders = [dict(steps={'label':genre_list})]
fig_genre_freq.update_layout(sliders=sliders)

##################################################################################
### Plot: Lineplot: The mean rating per genre by year
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_rating_genre_year(movies):
    return wrangle_rating_genre_year(movies)
mean_rating_genre_year = st_wrangle_rating_genre_year(movies)

fig_mean_rating_genre = go.Figure()
# Genre rating by year
mygenre_list = ['Action','Adventure','Comedy','Crime','Drama','Fantasy','Horror','Romance', 'Sci-Fi', 'Thriller']
for genre in mygenre_list:
	df = mean_rating_genre_year[mean_rating_genre_year['myGenre']==genre]
	fig_mean_rating_genre.add_trace(go.Scatter(x=df.startYear, y=df.meanRating,
						name = genre, yaxis='y', mode="markers+lines", customdata=np.stack(df.myGenre, axis=-1),
						hovertemplate="<br>".join([
							"Year: %{x}",
							"Mean rating: %{y}",
							"Genre: %{customdata}",
							"<extra></extra>" # remove 2nd box
						])))

# Create axis objects
fig_mean_rating_genre.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean IMDB Rating",
        range=[4, 9],
		#titlefont=dict(color="#1f77b4"),
		#tickfont=dict(color="#1f77b4")
		),
	xaxis=dict(title="Year")
)

fig_mean_rating_genre.update_layout(
	title_text="Mean Rating per Genre by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.12, xanchor="right", x=0.94, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
##############################################################################################
##################################################################################
### Plot: Lineplot: The mean runtime per genre by year
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_runtime_genre_year(movies):
    return wrangle_runtime_genre_year(movies)
mean_runtime_genre_year = st_wrangle_runtime_genre_year(movies)

fig_mean_runtime_genre = go.Figure()
# Genre runtime by year
mygenre_list = ['Action','Adventure','Comedy','Crime','Drama','Fantasy','Horror','Romance', 'Sci-Fi', 'Thriller']
for genre in mygenre_list:
	df = mean_runtime_genre_year[mean_runtime_genre_year['myGenre']==genre]
	fig_mean_runtime_genre.add_trace(go.Scatter(x=df.startYear, y=df.meanRuntime,
						name = genre, yaxis='y', mode="markers+lines", customdata=np.stack(df.myGenre, axis=-1),
						hovertemplate="<br>".join([
							"Year: %{x}",
							"Mean runtime [min]: %{y}",
							"Genre: %{customdata}",
							"<extra></extra>" # remove 2nd box
						])))

# Create axis objects
fig_mean_runtime_genre.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean Runtime [Minutes]",
        range=[50, 180],
		#titlefont=dict(color="#1f77b4"),
		#tickfont=dict(color="#1f77b4")
		),
	xaxis=dict(title="Year")
)

fig_mean_runtime_genre.update_layout(
	title_text="Mean Runtime per Genre by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.12, xanchor="right", x=0.94, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)

##################################################################################
### Plot: Lineplot: Mean budget, mean revenue, most expensive movie per year
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_budget_revenue_movies_lineplot(movies):
    return wrangle_budget_revenue_movies_lineplot(movies)

mean_budget_year, mean_revenue_year, max_revenue_year, max_budget_year = st_wrangle_budget_revenue_movies_lineplot(movies)

fig_financial = go.Figure()
# mean budget by year
fig_financial.add_trace(go.Scatter(x=mean_budget_year.index, y=mean_budget_year,
					name = 'Mean budget', yaxis='y', mode="markers+lines",
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean budget: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# mean revenue by year
fig_financial.add_trace(go.Scatter(x=mean_revenue_year.index, y=mean_revenue_year,
					name = 'Mean revenue', yaxis='y', mode="markers+lines",
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean revenue: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# max budget by year
fig_financial.add_trace(go.Scatter(x=max_budget_year.startYear, y=max_budget_year.budget,
					name = 'Maximum budget', yaxis='y', mode="markers+lines", customdata=np.stack(max_budget_year.primaryTitle, axis=-1),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Max. budget: %{y}",
						"Film: %{customdata}",
						"<extra></extra>" # remove 2nd box
                    ])))
# max revenue by year
fig_financial.add_trace(go.Scatter(x=max_revenue_year.startYear, y=max_revenue_year.revenue,
					name = 'Maximum revenue', yaxis='y2', mode="markers+lines", customdata=np.stack(max_revenue_year.primaryTitle, axis=-1), line=dict(color="purple"),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Max. revenue: %{y}",
						"Film: %{customdata}",
						"<extra></extra>" # remove 2nd box
                    ])))


# Create axis objects
fig_financial.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Dollars $",
		titlefont=dict(color="#1f77b4"),
		tickfont=dict(color="#1f77b4"),
        showgrid=False),
				
	#create 2nd y axis	
	yaxis2=dict(title="Max. revenue [$]",overlaying="y",
				side="right",
				titlefont=dict(color="purple"),
				tickfont=dict(color="purple"),
                showgrid=False),
	xaxis=dict(title="Year")
)

fig_financial.update_layout(
	title_text="Budget and Revenue by Year (not adjusted for inflation)",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.13, xanchor="right", x=0.99, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)

##############################################################################################
# budget vs year
# most rated movie vs year
# ereleases nby year#
### Let user select film by genome tags
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_genome_data():
    return load_genome_data()

@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_pure_tags():
    return load_pure_tags()

genome = st_load_genome_data()  # this is filtered to relevance >= 0.5
pure_tags = st_load_pure_tags()  # this is unfiltered

tag_list_total = genome['tag'].unique()
tag_list_user = st.multiselect(label='Choose your film tags:',
						options=tag_list_total,
						default="great ending",
						max_selections=10
)

# calculate total relevance combining all user-input tags
user_subset_tags = pure_tags[pure_tags['tag'].isin(tag_list_user)]
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
user_tag_recommendations = user_tag_recommendations[['imdbId','meanRelevance','primaryTitle','averageRating','numVotes','startYear','runtimeMinutes','genres','directors','tagline']]
user_tag_recommendations = user_tag_recommendations.set_index('imdbId', drop=True)
user_tag_recommendations = user_tag_recommendations.rename(columns={'meanRelevance':'Mean Relevance', 'primaryTitle':'Title', 'startYear':'Year', 'runtimeMinutes':'Runtime [Min]', 'genres':'Genres', 'averageRating':'IMDB Rating', 'numVotes':'IMDB Votes', 'directors':'Director', 'tagline':'Tagline'})
user_tag_recommendations = user_tag_recommendations.sort_values(by='Mean Relevance', ascending=False)

st.subheader('Top 50 results matching your tags by mean relevance:')
user_tag_recommendations

st.plotly_chart(fig_highest_rated)
st.plotly_chart(fig_financial)
st.plotly_chart(fig_genre_freq)
st.plotly_chart(fig_mean_rating_genre)
st.plotly_chart(fig_mean_runtime_genre)

