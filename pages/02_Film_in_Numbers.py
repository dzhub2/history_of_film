import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from  plotting_functions import *

### Define page-wide parameters
BGCOLOR = 'lightskyblue'
FONTSIZE = 15
FONTFAMILY = 'Garamond'
PERSIST = False
CACHESUPPRESS = True
MARKERLINEWIDTH = 2 #2
MARKERSIZE = 6.5
MARKEREDGECOLOR = 'midnightblue'

st.sidebar.markdown(":roller_coaster: Film in Numbers :roller_coaster:")

### Load and wrangle dataset
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_movie_data_from_genome():
    return load_movie_data_from_genome()
movies = st_load_movie_data_from_genome()

### Start the Page
_, col2, _ = st.columns([0.1, 0.5, 0.1])
with col2:
    st.title(":roller_coaster: Film in Numbers :roller_coaster:")

_, col2, _ = st.columns([0.1, 0.25, 0.1])
with col2:
    st.video("https://www.youtube.com/watch?v=2-qrMz-JAzo")

st.write("This page showcases the evolution of film throughout the years using different metrics like ratings, box office returns, etc. \
The data is mostly based on the official [IMDB datasets](https://www.imdb.com/interfaces/), limited to the subset of films contained in the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/). Information on budget and revenue was obtained from [TMDB](https://www.themoviedb.org/)")

st.write("**Note:** All graphics are **interactive**. You can hover the cursor, zoom in and select elements. (Double) click on elements in the legend to focus on the data you are most interested in!")
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
					name = 'Best rating', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					customdata=np.stack(max_rating_year.primaryTitle, axis=-1),
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Title: %{customdata}",
                        "Year: %{x}",
                        "IMDB Rating: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# mean rating per year
fig_highest_rated.add_trace(go.Scatter(x=avg_rating_per_year.index, y=avg_rating_per_year.meanAverageRating,
					name = 'Mean rating', mode="markers+lines",
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fe6c6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean rating: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# mean votes per year
fig_highest_rated.add_trace(go.Scatter(x=norm_votes_per_year.index, y=norm_votes_per_year.averageNrRatings,
					name = 'Mean votes', yaxis="y2", mode="markers+lines",
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#7e75ff", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Mean votes: %{y}",
                        "Year: %{x}"
						"<extra></extra>"
                    ])))
# max votes per year
fig_highest_rated.add_trace(go.Scatter(x=max_votes_year.startYear, y=max_votes_year.numVotes/10,
					name = 'Max. votes/10', yaxis="y2", mode="markers+lines", customdata=np.stack(max_votes_year.primaryTitle, axis=-1),
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#ff75b0", font=dict(color='white')),
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
		titlefont=dict(color="#3fa4e6", size=FONTSIZE),
		tickfont=dict(color="#3fa4e6", size=FONTSIZE),
        showgrid=True,
		range=[5.99, 9.6],
	),
					
		#create 2nd y axis	
		yaxis2=dict(title="Votes",overlaying="y",
					side="right",
					titlefont=dict(color="#ff75b0", size=FONTSIZE),
					tickfont=dict(color="#ff75b0", size=FONTSIZE),
					showgrid=False,
					range=[0, 300000]
	),
		xaxis=dict(title="Year", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE),),
)

fig_highest_rated.update_layout(
	title_text="Ratings and Votes by Year",#	width=800
	legend_font_size = FONTSIZE,
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=1, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)

fig_highest_rated.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
# line colors
fig_highest_rated.data[0].line.color = "#3fa4e6"
fig_highest_rated.data[1].line.color = "#3fe6c6"
fig_highest_rated.data[2].line.color = "#7e75ff"
fig_highest_rated.data[3].line.color = "#ff75b0"

##################################################################################
### Plot: Bar: The number of movies (in percent) per year as an animation by genre
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_count_genre_year_bar(movies, nr_movies_per_year):
    return wrangle_count_genre_year_bar(movies, nr_movies_per_year)
genres_bar_percent, genre_list = st_wrangle_count_genre_year_bar(movies, nr_movies_per_year)
genres_bar_percent = genres_bar_percent.sort_values('genre')

fig_genre_freq = px.bar(genres_bar_percent, x='startYear', y='count', color_discrete_sequence =['#ff75b0']*len(genres_bar_percent),
    animation_frame=genres_bar_percent.genre, range_y=[0,82], title='Percentage of Genre per Year',
	hover_data=['startYear', 'count'], labels={'startYear':'Year', 'count':'Percent', 'genre':'Genre'})

fig_genre_freq.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000 # animation transition speed

myx = movies['startYear'].sort_values().unique()[0::2] # only plot every 2nd year
#fig_genre_freq.update_traces(textfont_size=FONTSIZE, textfont_family= FONTFAMILY, textangle=0, textposition="outside", cliponaxis=False)
fig_genre_freq.update_xaxes(tickmode='array', tickangle=75, title='Year', tickvals=myx)
fig_genre_freq.update_yaxes(tickmode='auto', tickangle=0, title='Percent')

sliders = [dict(steps={'label':genre_list})]
fig_genre_freq.update_layout(sliders=sliders)
fig_genre_freq.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
fig_genre_freq.update_layout(
    font_family=FONTFAMILY,
    font_size=FONTSIZE-2,
    plot_bgcolor = 'rgba(0, 0, 0, 0)',
	yaxis = dict(
	titlefont=dict(size=FONTSIZE),
	tickfont=dict(size=FONTSIZE),
	),
	xaxis=dict(titlefont=dict(size=FONTSIZE),
	tickfont=dict(size=FONTSIZE),),
)
##################################################################################
### Plot: Lineplot: The mean rating per genre by year
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_rating_genre_year(movies):
    return wrangle_rating_genre_year(movies)
mean_rating_genre_year = st_wrangle_rating_genre_year(movies)

fig_mean_rating_genre = go.Figure()
# Genre rating by year
mygenre_list = ['Animation','Action','Adventure','Comedy','Crime','Drama','Fantasy','Horror','Mystery', 'Romance', 'Sci-Fi', 'Thriller']
for genre in mygenre_list:
	df = mean_rating_genre_year[mean_rating_genre_year['myGenre']==genre]
	fig_mean_rating_genre.add_trace(go.Scatter(x=df.startYear, y=df.meanRating,
						name = genre, yaxis='y', mode="markers+lines", customdata=np.stack(df.myGenre, axis=-1),
						marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
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
        range=[3.9, 9.1],
		titlefont=dict( size=FONTSIZE),
		tickfont=dict( size=FONTSIZE)
		),
	xaxis=dict(title="Year", titlefont=dict( size=FONTSIZE),
		tickfont=dict( size=FONTSIZE))
)

fig_mean_rating_genre.update_layout(
	title_text="Mean Rating per Genre by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=1, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_mean_rating_genre.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
##################################################################################
### Plot: Lineplot: The mean runtime per genre by year
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_runtime_genre_year(movies):
    return wrangle_runtime_genre_year(movies)
mean_runtime_genre_year = st_wrangle_runtime_genre_year(movies)

fig_mean_runtime_genre = go.Figure()
# Genre runtime by year
mygenre_list = ['Animation','Action','Adventure','Comedy','Crime','Drama','Fantasy','Horror','Mystery', 'Romance', 'Sci-Fi', 'Thriller']
for genre in mygenre_list:
	df = mean_runtime_genre_year[mean_runtime_genre_year['myGenre']==genre]
	fig_mean_runtime_genre.add_trace(go.Scatter(x=df.startYear, y=df.meanRuntime,
						name = genre, yaxis='y', mode="markers+lines", customdata=np.stack(df.myGenre, axis=-1),
						marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
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
		title="Mean Runtime [minutes]",
        range=[50, 170],
		titlefont=dict( size=FONTSIZE),
		tickfont=dict( size=FONTSIZE)
		),
	xaxis=dict(title="Year", titlefont=dict( size=FONTSIZE),
		tickfont=dict( size=FONTSIZE))
)

fig_mean_runtime_genre.update_layout(
	title_text="Mean Runtime per Genre by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=1, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_mean_runtime_genre.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
##################################################################################
### Plot: Lineplot: The mean revenue per genre by year
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_wrangle_revenue_genre_year(movies):
    return wrangle_revenue_genre_year(movies)
mean_revenue_genre_year = st_wrangle_revenue_genre_year(movies)

fig_mean_revenue_genre = go.Figure()
# Genre revenue by year
mygenre_list = ['Animation','Action','Adventure','Comedy','Crime','Drama','Fantasy','Horror','Mystery', 'Romance', 'Sci-Fi', 'Thriller']
for genre in mygenre_list:
	df = mean_revenue_genre_year[mean_revenue_genre_year['myGenre']==genre]
	fig_mean_revenue_genre.add_trace(go.Scatter(x=df.startYear, y=df.meanRevenue,
						name = genre, yaxis='y', mode="markers+lines", customdata=np.stack(df.myGenre, axis=-1),
						marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH-0.5, marker_size=MARKERSIZE-0.5,
						hovertemplate="<br>".join([
							"Year: %{x}",
							"Mean revenue [$]: %{y}",
							"Genre: %{customdata}",
							"<extra></extra>" # remove 2nd box
						])))

# Create axis objects
fig_mean_revenue_genre.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean Revenue in $",
        range=[-50000000, 700e6],
		titlefont=dict( size=FONTSIZE),
		tickfont=dict( size=FONTSIZE)
		),
	xaxis=dict(title="Year", titlefont=dict( size=FONTSIZE),
		tickfont=dict( size=FONTSIZE))
)

fig_mean_revenue_genre.update_layout(
	title_text="Mean Revenue per Genre by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=1, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_mean_revenue_genre.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
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
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean budget: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# mean revenue by year
fig_financial.add_trace(go.Scatter(x=mean_revenue_year.index, y=mean_revenue_year,
					name = 'Mean revenue', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fe6c6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean revenue: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# max budget by year
fig_financial.add_trace(go.Scatter(x=max_budget_year.startYear, y=max_budget_year.budget,
					name = 'Maximum budget', yaxis='y', mode="markers+lines", customdata=np.stack(max_budget_year.primaryTitle, axis=-1),
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#7e75ff", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Max. budget: %{y}",
						"Film: %{customdata}",
						"<extra></extra>" # remove 2nd box
                    ])))
# max revenue by year
fig_financial.add_trace(go.Scatter(x=max_revenue_year.startYear, y=max_revenue_year.revenue,
					name = 'Maximum revenue', yaxis='y2', mode="markers+lines", customdata=np.stack(max_revenue_year.primaryTitle, axis=-1),
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#ff75b0", font=dict(color='white')),
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
		title="Budget/Revenue [$]",
		titlefont=dict(color="#1f77b4",size=FONTSIZE),
		tickfont=dict(color="#1f77b4",size=FONTSIZE),
        showgrid=True),
				
	#create 2nd y axis	
	yaxis2=dict(title="Max. revenue [$]",overlaying="y",
				side="right",
				titlefont=dict(color="#ff75b0",size=FONTSIZE),
				tickfont=dict(color="#ff75b0",size=FONTSIZE),
                showgrid=False),
	xaxis=dict(title="Year", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE))
)

fig_financial.update_layout(
	title_text="Budget and Revenue by Year (not adjusted for inflation)",	width=800,
	legend_font_size = FONTSIZE,
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=1, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_financial.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)

# line colors
fig_financial.data[0].line.color = "#3fa4e6"
fig_financial.data[1].line.color = "#3fe6c6"
fig_financial.data[2].line.color = "#7e75ff"
fig_financial.data[3].line.color = "#ff75b0"

### Print the plots

st.plotly_chart(fig_highest_rated)
st.plotly_chart(fig_genre_freq)
st.plotly_chart(fig_mean_rating_genre)
st.plotly_chart(fig_mean_runtime_genre)
st.plotly_chart(fig_mean_revenue_genre)
st.plotly_chart(fig_financial)

##################################################################################
### Plot: Lineplot: Biggest Box Office Hits and Flops
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_return_by_year(movies):
    return return_by_year(movies)

max_return_by_year, min_return_by_year = st_return_by_year(movies)

fig_return = go.Figure()
# max return by year
fig_return.add_trace(go.Scatter(x=max_return_by_year.startYear, y=max_return_by_year['return'],
					name = 'Highest return on investment (Hit)', yaxis='y', mode="markers+lines", customdata=np.stack((max_return_by_year['primaryTitle'],max_return_by_year['budget'],max_return_by_year['revenue']), axis=-1),
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Max. return: %{y}",
						"Film: %{customdata[0]}",
						"Budget [$]: %{customdata[1]}",
						"Revenue [$]: %{customdata[2]}",
						"<extra></extra>" # remove 2nd box
                    ])))
# min return by year
fig_return.add_trace(go.Scatter(x=min_return_by_year.startYear, y=min_return_by_year['return'],
					name = 'Most money lost (Flop)', yaxis='y2', mode="markers+lines", customdata=np.stack((min_return_by_year['primaryTitle'],min_return_by_year['budget'],min_return_by_year['revenue']), axis=-1),
					marker_symbol='circle', marker_line_color=MARKEREDGECOLOR, marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#ff75b0", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Min. return: %{y}",
						"Film: %{customdata[0]}",
						"Budget [$]: %{customdata[1]}",
						"Revenue [$]: %{customdata[2]}",
						"<extra></extra>" # remove 2nd box
                    ])))

# Create axis objects
fig_return.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Hits - Return on Investment [multiple of budget]",
		titlefont=dict(color="#3fa4e6",size=FONTSIZE),
		tickfont=dict(color="#3fa4e6",size=FONTSIZE),
        showgrid=True,
		range=[-150, 5000]),	
	#create 2nd y axis	
	yaxis2=dict(title="Flops - Return on Investment [multiple of budget]",overlaying="y",
				side="right",
				titlefont=dict(color="#ff75b0",size=FONTSIZE),
				tickfont=dict(color="#ff75b0",size=FONTSIZE),
                showgrid=False,
				range=[-1.1, 0]),
	xaxis=dict(title="Year", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE),)
)

fig_return.update_layout(
	title_text="Biggest Hits and Box Office Flops by Year",#	width=800
	legend_font_size = FONTSIZE,
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.13, xanchor="right", x=0.99, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_return.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)

# line colors
fig_return.data[0].line.color = "#3fa4e6"
fig_return.data[1].line.color = "#ff75b0"

st.plotly_chart(fig_return)


##################################################################################
### Franchise Information
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_franchises():
    return load_franchises()

franchises_pivot = st_load_franchises()
franchises_pivot.style.format('{:.2f}')

_, col2, _ = st.columns([0.75, 6, 1.0])
with col2:
	st.subheader("Top 15 Most Profitable Film Franchises")

_, col2, _ = st.columns([1.0, 4, 1])
with col2:
	df = franchises_pivot.sort_values('Total [$]', ascending=False, ignore_index=True)
	df['Mean [$]'] = df['Mean [$]']/1000000 # convert to M
	df['Total [$]'] = df['Total [$]']/1000000 # convert to M
	df = df.iloc[0:15]
	st.dataframe(df.style.format({'Mean [$]' : "{:.2f}M", 'Total [$]' : "{:.2f}M"}))

_, col2, _ = st.columns([0.65, 6, 1.0])
with col2:
	st.subheader("Top 20 Longest Running Film Franchises")

_, col2, _ = st.columns([1.0, 4, 1])
with col2:
	df = franchises_pivot.sort_values('Films', ascending=False, ignore_index=True)
	df['Mean [$]'] = df['Mean [$]']/1000000 # convert to M
	df['Total [$]'] = df['Total [$]']/1000000 # convert to M
	df = df.iloc[0:20]
	st.dataframe(df.style.format({'Mean [$]' : "{:.2f}M", 'Total [$]' : "{:.2f}M"}))

