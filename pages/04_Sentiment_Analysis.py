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
MARKERLINEWIDTH = 2
MARKERSIZE = 6


st.sidebar.markdown(":performing_arts: Sentiment Analysis :performing_arts:")

_, col2, _ = st.columns([0.2, 1.3, 0.2])
with col2:
    st.title(":performing_arts: Sentiment Analysis:performing_arts:")

_, col2, _ = st.columns([0.1, 0.25, 0.1])
with col2:
    st.video("https://www.youtube.com/watch?v=FnpJBkAMk44")

st.header("Film Keyword Sentiment Analysis")

st.write("The Movie Database ([TMDB](https://www.themoviedb.org/)) offers several keywords associated with \
the plot and general themes of a film. Sentiment Analysis for ~12.000 films was performed on these keywords using \
the NLP framework [flair](https://github.com/flairNLP/flair). This library is particularly well \
suited as it was pre-trained on IMDB data.")
# here, use dialogue kaggle data to perform sentiment analysis on

# @st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
# def st_load_dialogue_data():
#     return load_dialogue_data()

# title_corpus = st_load_dialogue_data()

# @st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
# def st_dialogue_genre_scores(title_corpus):
#     return dialogue_genre_scores(title_corpus)

# genre_list, pos_score_ratings, neg_score_ratings, pos_score_count, neg_score_count,\
# pos_votes_ratings, neg_votes_ratings  = st_dialogue_genre_scores(title_corpus)

# ##################################################################################
# ### Plot: Lineplot: Mean IMDB rating per genre pos. vs. neg. sentiment
# fig_rating_sentiment = go.Figure()
# # pos. sentiment rating
# fig_rating_sentiment.add_trace(go.Scatter(x=genre_list, y=pos_score_ratings,
# 					name = 'Positive sentiment', yaxis='y', mode="markers+lines",
# 					hovertemplate="<br>".join([
#                         "Genre: %{x}",
#                         "IMDB Rating: %{y}",
# 						"<extra></extra>" # remove 2nd box
#                     ])))
# # neg. sentiment rating
# fig_rating_sentiment.add_trace(go.Scatter(x=genre_list, y=neg_score_ratings,
# 					name = 'Negative sentiment', yaxis='y', mode="markers+lines",
# 					hovertemplate="<br>".join([
#                         "Genre: %{x}",
#                         "IMDB Rating: %{y}",
# 						"<extra></extra>" # remove 2nd box
#                     ])))

# # Create axis objects
# fig_rating_sentiment.update_layout(
# 	font_family = FONTFAMILY,
# 	font_size = FONTSIZE,
# 	#create 1st y axis			
# 	yaxis=dict(
# 		title="Mean IMDB Rating",
# 		titlefont=dict(color="#1f77b4"),
# 		tickfont=dict(color="#1f77b4"),
#         showgrid=True),
# 	xaxis=dict(title="Genre")
# )
# fig_rating_sentiment.update_xaxes(tickangle=65)

# fig_rating_sentiment.update_layout(
# 	title_text="Mean IMDB Rating per Genre: Positive vs. Negative Dialogue Sentiment",#	width=800
# 	hovermode="x", # or just x
# 	plot_bgcolor = 'rgba(0, 0, 0, 0)',
# 	legend=dict(yanchor="top", y=1.27, xanchor="right", x=0.9, orientation="h",
# 	bgcolor="white", bordercolor="Black", borderwidth=1)
# )

# ##################################################################################
# ### Plot: Lineplot: Mean number of votes genre pos. vs. neg. sentiment
# fig_votes_sentiment = go.Figure()
# # pos. sentiment rating
# fig_votes_sentiment.add_trace(go.Scatter(x=genre_list, y=pos_votes_ratings,
# 					name = 'Positive sentiment', yaxis='y', mode="markers+lines",
# 					hovertemplate="<br>".join([
#                         "Genre: %{x}",
#                         "Votes: %{y}",
# 						"<extra></extra>" # remove 2nd box
#                     ])))
# # neg. sentiment rating
# fig_votes_sentiment.add_trace(go.Scatter(x=genre_list, y=neg_votes_ratings,
# 					name = 'Negative sentiment', yaxis='y', mode="markers+lines",
# 					hovertemplate="<br>".join([
#                         "Genre: %{x}",
#                         "Votes: %{y}",
# 						"<extra></extra>" # remove 2nd box
#                     ])))

# # Create axis objects
# fig_votes_sentiment.update_layout(
# 	font_family = FONTFAMILY,
# 	font_size = FONTSIZE,
# 	#create 1st y axis			
# 	yaxis=dict(
# 		title="Mean IMDB Votes",
# 		titlefont=dict(color="#1f77b4"),
# 		tickfont=dict(color="#1f77b4"),
#         showgrid=True),
# 	xaxis=dict(title="Genre")
# )
# fig_votes_sentiment.update_xaxes(tickangle=65)

# fig_votes_sentiment.update_layout(
# 	title_text="Mean IMDB Votes per Genre: Positive vs. Negative Dialogue Sentiment",#	width=800
# 	hovermode="x", # or just x
# 	plot_bgcolor = 'rgba(0, 0, 0, 0)',
# 	legend=dict(yanchor="top", y=1.27, xanchor="right", x=0.9, orientation="h",
# 	bgcolor="white", bordercolor="Black", borderwidth=1)
# )

# ##################################################################################
# ### Plot: Bar: film-count for neg and pos sentiment per genre
# dfpos = pd.concat([pd.Series(pos_score_count), pd.Series(genre_list)], axis=1)
# dfpos.columns = ['count','genre']
# dfneg = pd.concat([pd.Series(neg_score_count), pd.Series(genre_list)], axis=1)
# dfneg.columns = ['count','genre']

# fig_sentiment_genre_count = go.Figure()
# fig_sentiment_genre_count.add_trace(go.Bar(x=dfpos['genre'], y=dfpos['count'],
#                 name='Positive sentiment', offsetgroup=0))
# fig_sentiment_genre_count.add_trace(go.Bar(x=dfneg['genre'], y=dfneg['count'],
#                 name='Negative sentiment', offsetgroup=1))

# fig_sentiment_genre_count.update_layout(
#     xaxis=dict(
#         showgrid=False,
#     ),
#     yaxis=dict(
# 		title='Number of Films',
#         showgrid=True,
#         range=[0, 354]
#     ),
# 	font=dict(
#         family=FONTFAMILY,
#         size=FONTSIZE,
#     ),
#     barmode='stack',
# )
# fig_sentiment_genre_count.update_layout(
# 	title_text="Positive vs. Negative Sentiment Count per Genre",#	width=800
# 	hovermode="x", # or just x
# 	plot_bgcolor = 'rgba(0, 0, 0, 0)',
# 	legend=dict(yanchor="top", y=1.12, xanchor="right", x=0.94, orientation="h",
# 	bgcolor="white", bordercolor="Black", borderwidth=1),
# 	xaxis=dict(title="Genre")
# )
# fig_sentiment_genre_count.update_xaxes(tickangle=65)
# #fig_sentiment_genre_count.layout.font.family = 'Arial'

# st.plotly_chart(fig_rating_sentiment)
# st.plotly_chart(fig_votes_sentiment)
# st.plotly_chart(fig_sentiment_genre_count)

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_keyword_data():
    return load_keyword_data()

genome_movies_sentiment = st_load_keyword_data()

@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_keyword_genre_scores(genome_movies_sentiment):
    return keyword_genre_scores(genome_movies_sentiment)

genre_list, pos_score_ratings, neg_score_ratings, pos_score_count,\
neg_score_count, pos_votes_ratings, neg_votes_ratings, mean_score_year  = st_keyword_genre_scores(genome_movies_sentiment)

##################################################################################
### Plot: Lineplot: Mean IMDB rating per genre pos. vs. neg. sentiment
fig_rating_sentiment = go.Figure()
# pos. sentiment rating
fig_rating_sentiment.add_trace(go.Scatter(x=genre_list, y=pos_score_ratings,
					name = 'Positive sentiment', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color="midnightblue", marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Genre: %{x}",
                        "IMDB Rating: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# neg. sentiment rating
fig_rating_sentiment.add_trace(go.Scatter(x=genre_list, y=neg_score_ratings,
					name = 'Negative sentiment', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color="midnightblue", marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#ff75b0", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Genre: %{x}",
                        "IMDB Rating: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))

# Create axis objects
fig_rating_sentiment.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean IMDB Rating",
		titlefont=dict(color="#1f77b4", size=FONTSIZE),
		tickfont=dict(color="#1f77b4", size=FONTSIZE),
        showgrid=True,
		range=[5.4, 8.1]),
	xaxis=dict(title="Genre", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE))
)
fig_rating_sentiment.update_xaxes(tickangle=65)

fig_rating_sentiment.update_layout(
	title_text="Mean IMDB Rating per Genre: Positive vs. Negative Sentiment",#	width=800
	legend_font_size = FONTSIZE,
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=0.97, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_rating_sentiment.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
# line colors
fig_rating_sentiment.data[0].line.color = "#3fa4e6"
fig_rating_sentiment.data[1].line.color = "#ff75b0"

##################################################################################
### Plot: Lineplot: Mean number of votes genre pos. vs. neg. sentiment
fig_votes_sentiment = go.Figure()
# pos. sentiment rating
fig_votes_sentiment.add_trace(go.Scatter(x=genre_list, y=pos_votes_ratings,
					name = 'Positive sentiment', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color="midnightblue", marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Genre: %{x}",
                        "Votes: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))
# neg. sentiment rating
fig_votes_sentiment.add_trace(go.Scatter(x=genre_list, y=neg_votes_ratings,
					name = 'Negative sentiment', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color="midnightblue", marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#ff75b0", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Genre: %{x}",
                        "Votes: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))

# Create axis objects
fig_votes_sentiment.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean IMDB Votes",
		titlefont=dict(color="#1f77b4", size=FONTSIZE),
		tickfont=dict(color="#1f77b4", size=FONTSIZE),
        showgrid=True,
		range=[0, 201000]),
	xaxis=dict(title="Genre", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE))
)
fig_votes_sentiment.update_xaxes(tickangle=65)

fig_votes_sentiment.update_layout(
	title_text="Mean IMDB Votes per Genre: Positive vs. Negative Sentiment",#	width=800
	legend_font_size = FONTSIZE,
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=0.97, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_votes_sentiment.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
# line colors
fig_votes_sentiment.data[0].line.color = "#3fa4e6"
fig_votes_sentiment.data[1].line.color = "#ff75b0"

##################################################################################
### Plot: Lineplot: Mean sentiment score by year
fig_sentiment_year = go.Figure()
# pos. sentiment rating
fig_sentiment_year.add_trace(go.Scatter(x=mean_score_year.index, y=mean_score_year,
					name = 'Mean sentiment', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color="midnightblue", marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean sentiment: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))

# Create axis objects
fig_sentiment_year.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean Sentiment Score",
		titlefont=dict(color="#1f77b4", size=FONTSIZE),
		tickfont=dict(color="#1f77b4", size=FONTSIZE),
        showgrid=True,
		range=[-0.61, 0.41]),
	xaxis=dict(title="Year", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE))
)
fig_sentiment_year.update_xaxes(tickangle=65)

fig_sentiment_year.update_layout(
	title_text="Mean Sentiment Score by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.27, xanchor="right", x=0.9, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_sentiment_year.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
# line colors
fig_sentiment_year.data[0].line.color = "#3fa4e6"

##################################################################################
### Plot: Bar: film-count for neg and pos sentiment per genre
dfpos = pd.concat([pd.Series(pos_score_count), pd.Series(genre_list)], axis=1)
dfpos.columns = ['count','genre']
dfneg = pd.concat([pd.Series(neg_score_count), pd.Series(genre_list)], axis=1)
dfneg.columns = ['count','genre']

fig_sentiment_genre_count = go.Figure()
fig_sentiment_genre_count.add_trace(go.Bar(x=dfpos['genre'], y=dfpos['count'],
                name='Positive sentiment', offsetgroup=0,  marker_color='#3fa4e6'))
fig_sentiment_genre_count.add_trace(go.Bar(x=dfneg['genre'], y=dfneg['count'],
                name='Negative sentiment', offsetgroup=1,  marker_color='#ff75b0'))

fig_sentiment_genre_count.update_layout(
    xaxis=dict(
        showgrid=False,
    ),
    yaxis=dict(
		title='Number of Films',
        showgrid=True,
		titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE),
       	range=[0, 7050]
    ),
	font=dict(
        family=FONTFAMILY,
        size=FONTSIZE,
    ),
    barmode='stack',
)
fig_sentiment_genre_count.update_layout(
	title_text="Positive vs. Negative Sentiment Count per Genre",#	width=800
	legend_font_size = FONTSIZE,
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.14, xanchor="right", x=0.97, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1),
	xaxis=dict(title="Genre", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE))
)
fig_sentiment_genre_count.update_xaxes(tickangle=65)
#fig_sentiment_genre_count.layout.font.family = 'Arial'
fig_sentiment_genre_count.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)

st.plotly_chart(fig_rating_sentiment)
st.plotly_chart(fig_votes_sentiment)
st.plotly_chart(fig_sentiment_year)
st.plotly_chart(fig_sentiment_genre_count)


st.header("Film Dialogue Sentiment Analysis")
st.write("For comparison to the somewhat simplified keyword analysis above, the \
[Cornell Movie--Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) \
was also analysed. However, it contains only ~600 films.")

@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_dialogue_data_mean():
    return load_dialogue_data_mean()

title_corpus_and_mean = st_load_dialogue_data_mean()
target = title_corpus_and_mean.groupby('year').mean()['sentiment_score']
##################################################################################
### Plot: Lineplot: Mean sentiment score by year
fig_sentiment_year = go.Figure()
# pos. sentiment rating
fig_sentiment_year.add_trace(go.Scatter(x=target.index, y=target,
					name = 'Mean sentiment', yaxis='y', mode="markers+lines",
					marker_symbol='circle', marker_line_color="midnightblue", marker_line_width=MARKERLINEWIDTH, marker_size=MARKERSIZE,
					hoverlabel=dict(bgcolor= "#3fa4e6", font=dict(color='white')),
					hovertemplate="<br>".join([
                        "Year: %{x}",
                        "Mean sentiment: %{y}",
						"<extra></extra>" # remove 2nd box
                    ])))

# Create axis objects
fig_sentiment_year.update_layout(
	font_family = FONTFAMILY,
	font_size = FONTSIZE,
	#create 1st y axis			
	yaxis=dict(
		title="Mean Sentiment Score",
		titlefont=dict(color="#1f77b4", size=FONTSIZE),
		tickfont=dict(color="#1f77b4", size=FONTSIZE),
        showgrid=True,
		range=[-0.61, 0.41]),
	xaxis=dict(title="Year", titlefont=dict(size=FONTSIZE),
		tickfont=dict(size=FONTSIZE))
)
fig_sentiment_year.update_xaxes(tickangle=65)

fig_sentiment_year.update_layout(
	title_text="Mean Sentiment Score by Year",#	width=800
	hovermode="x", # or just x
	plot_bgcolor = 'rgba(0, 0, 0, 0)',
	legend=dict(yanchor="top", y=1.27, xanchor="right", x=0.9, orientation="h",
	bgcolor="white", bordercolor="Black", borderwidth=1)
)
fig_sentiment_year.update_layout(
    hoverlabel=dict(
        font_size=FONTSIZE+0.5,
    )
)
# line colors
fig_sentiment_year.data[0].line.color = "#3fa4e6"