""" A collection of all functions used to extract and wrangle data in the main streamlin app.py"""

# imports
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
# def load_imdb_data():

#     movies_cleaned_directors_filtered = pd.read_csv("../output/movies_cleaned_directors_filtered.csv")
#     return movies_cleaned_directors_filtered

def load_movie_data_from_genome():
    "Load main genome DB and extract unique movies without tags"

    genome_imdb_ml_tmdb_cleaned_filtered = pd.read_csv("./output/genome_imdb_ml_tmdb_cleaned_filtered.csv")
    # drop the genome data and extract pure movie data
    movies = genome_imdb_ml_tmdb_cleaned_filtered.drop(columns=['relevance', 'tag'], axis=1)
    movies = movies.drop_duplicates(subset=['primaryTitle'])
    movies.reset_index(drop=True, inplace=True)

    return movies

def load_movie_filtered_director():
    "Load main genome DB, which has been cleaned, filtered and contains directors."

    movies_cleaned_directors_filtered = pd.read_csv("./output/movies_cleaned_directors_filtered.csv")
    
    return movies_cleaned_directors_filtered

def load_genome_data():
    "Load main genome DB inlcuding tags + relevance for movies from the movie lense DB."

    genome_imdb_ml_tmdb_cleaned_filtered = pd.read_csv("./output/genome_imdb_ml_tmdb_cleaned_filtered.csv")
    #genome_imdb_ml_tmdb_cleaned_filtered.reset_index(drop=True, inplace=True)
    return genome_imdb_ml_tmdb_cleaned_filtered

def load_pure_tags():
    "Load movie lens genome data without any prior filtering on the relevance of the tags."""

    genome_pure_tags = pd.read_csv("./output/genome_pure_tags.csv")
    return genome_pure_tags

def load_franchises():
    """Load the dataset to display various franchise metrics."""

    franchises_pivot = pd.read_csv("./output/franchises_pivot.csv")
    franchises_pivot = franchises_pivot[franchises_pivot['Mean [$]']>1000000] # filter high earning franchises
    return franchises_pivot

def load_dialogue_data():
    """Load the dataframe containing info about movies, dialogue and sentiment analysis."""

    title_corpus = pd.read_csv("./output/title_corpus.csv")
    return title_corpus

def load_keyword_data():
    """Load the dataframe containing info about movies, keywrods and sentiment analysis."""

    genome_movies_sentiment = pd.read_csv("./output/genome_movies_sentiment.csv")
    return genome_movies_sentiment

def load_quotes():
    """Load movie quote data for guessing game."""

    quotes = pd.read_csv("./output/quotes_cleaned.csv")
    quotes = quotes.sample(frac=1) # randomize the dataframe once at load-time
    return quotes

def load_regression_result():
    """Load XGBoost regression feature importance files."""

    importances_return = pd.read_csv("./output/importances_return.csv")
    importances_rating = pd.read_csv("./output/importances_rating.csv")

    return importances_return, importances_rating

def wrangle_highest_rated_movies_lineplot(movies):
    """Extract features for highest rated and votes movies by year plot"""

    max_rating_year = movies[['averageRating','numVotes', 'startYear','primaryTitle','imdbId']].sort_values(by='averageRating', ascending=False).drop_duplicates(['startYear']).sort_values(by='startYear')
    max_votes_year = movies[['averageRating','numVotes', 'startYear','primaryTitle','imdbId']].sort_values(by='numVotes', ascending=False).drop_duplicates(['startYear']).sort_values(by='startYear')
    nr_movies_per_year = movies[['averageRating', 'startYear']].groupby('startYear').count().rename(columns={'averageRating':'numMovies'})
    total_ratings_per_year = movies[['numVotes', 'startYear']].groupby('startYear').sum().rename(columns={'numVotes':'numVotesYear'})
    norm_votes_per_year = pd.DataFrame((total_ratings_per_year['numVotesYear']/nr_movies_per_year['numMovies']).round().astype('int32')).rename(columns={0:'averageNrRatings'})
    avg_rating_per_year = movies[['averageRating', 'startYear']].groupby('startYear').mean().rename(columns={'averageRating':'meanAverageRating'})

    return max_rating_year, nr_movies_per_year, total_ratings_per_year, norm_votes_per_year, avg_rating_per_year, max_votes_year

def wrangle_budget_revenue_movies_lineplot(movies):
    """Extract financial features for budget, revenue by year for line plot"""

    mean_budget_year = movies.groupby('startYear').mean()['budget']
    mean_revenue_year = movies.groupby('startYear').mean()['revenue']
    max_revenue_year = movies.sort_values(by='revenue', ascending=False).drop_duplicates(['startYear']).sort_values(by='startYear')
    max_budget_year = movies.sort_values(by='budget', ascending=False).drop_duplicates(['startYear']).sort_values(by='startYear')


    return mean_budget_year, mean_revenue_year, max_revenue_year, max_budget_year

def wrangle_count_genre_year_bar(movies, nr_movies_per_year):
    """Wrangle the movie dataframe to extract every genre of a movie and count how many movies
    per genre are released each year. Used for bar plot."""

    # genres from IMDB dataset are used
    # extract features and split genres into columns
    df = movies['genres'].str.split(',', expand=True)
    # find unique genres
    g1 = df[0].unique()
    g2 = df[1].unique()
    g3 = df[2].unique()
    g  = np.concatenate([g1,g2,g3])
    genre_list = pd.Series(g).unique()
    # remove nan values from list
    result = []
    for el in genre_list:
        if type(el) == str:
            result.append(el)
    genre_list = result
    # remove uninteresting genres
    try:
        genre_list.remove('Film-Noir')
    except:
        pass
    try:
        genre_list.remove('News')
    except:
        pass
    try:
        genre_list.remove('Short')
    except:
        pass
    try:
        genre_list.remove('Reality-TV')
    except:
        pass
    try:
        genre_list.remove('Sport')
    except:
        pass

    # combine all genres per year, then count how often each genre is contained in the created string
    genres_per_year = movies.groupby('startYear')['genres'].apply(lambda row: ','.join(row))
    genres_per_year_full = genres_per_year.str.count(genre_list[0])
    genres_per_year_full = pd.DataFrame(genres_per_year_full)
    for genre in genre_list[1:]:
        tmp = genres_per_year.str.count(genre)
        genres_per_year_full = pd.concat([genres_per_year_full, tmp], axis=1, ignore_index=True)
    # name columns by genre
    genres_per_year_full.columns = genre_list

    # wrangle final dataframe for bar plot
    genres_bar = pd.DataFrame()
    for idx in range(len(genre_list)):
        dm = pd.DataFrame(data={'startYear':genres_per_year_full.index, 'genre':genre_list[idx], 'count':genres_per_year_full[genre_list[idx]]})
        genres_bar = pd.concat([genres_bar,dm])

    # calculate percentage of each genre compared to total movies that year
    divider = nr_movies_per_year
    for k in range(len(genre_list)-1):
        divider = pd.concat([divider,nr_movies_per_year])
    genres_bar_percent = genres_bar
    genres_bar_percent['count'] = (genres_bar_percent['count']/divider['numMovies'])*100
    genres_bar_percent = genres_bar_percent.round(2)

    return genres_bar_percent, genre_list

def wrangle_rating_genre_year(movies):
    """Wrangle data for the line plot of avg. rating by genre over the years."""

    # the genres are listed in a single column inside a long string. First, we unwrap this string, by appending the whole dataframe at the 
    # bottom of itself over and over, each time collecting a single genre in a new "myGenre" column
    df = movies[['primaryTitle','startYear','genres','averageRating','numVotes']]
    genre_df = df['genres'].str.split(',', expand=True)
    max_nr_genres = len(genre_df.columns) # counts maximum number of genres listed per movie

    # do the appending after we extracted all individual genres as column
    genre_df[0].rename('myGenre', inplace=True)
    movies_genre_unfolded = pd.concat([df, genre_df[0]], axis = 1) # append genre as a column

    for k in range(1, max_nr_genres):
        genre_df[k].rename('myGenre', inplace=True)
        tmp = pd.concat([df, genre_df[k]], axis = 1) # append k'th genre as a column
        movies_genre_unfolded = pd.concat([movies_genre_unfolded, tmp], axis = 0) # append above dataframe at bottom of total frame
    movies_genre_unfolded.reset_index(drop=True, inplace=True)
    # clean
    movies_genre_unfolded.drop('genres', inplace=True, axis=1, errors='ignore')
    movies_genre_unfolded['myGenre'].fillna(value='no genre', inplace=True)
    # calculate the mean rating and clean
    mean_rating_genre_year = movies_genre_unfolded.groupby(by=['startYear','myGenre'])['averageRating'].mean().rename('meanRating', inplace=True)
    mean_rating_genre_year = pd.DataFrame(mean_rating_genre_year)
    mean_rating_genre_year.reset_index(inplace=True)
    mean_rating_genre_year = mean_rating_genre_year[mean_rating_genre_year['myGenre']!='no genre'] # drop no genre rows
    mean_rating_genre_year['meanRating'] = mean_rating_genre_year['meanRating'].round(2)
    # calculate the count of movies per genre per year
    count_genre_year = movies_genre_unfolded.groupby(by=['startYear','myGenre'])['primaryTitle'].count().rename('count', inplace=True)
    count_genre_year = pd.DataFrame(count_genre_year)
    count_genre_year.reset_index(inplace=True)
    count_genre_year = count_genre_year[count_genre_year['myGenre']!='no genre'] # drop no genre rows

    return mean_rating_genre_year

def wrangle_runtime_genre_year(movies):
    """Wrangle data for the line plot of avg. runtime by genre over the years."""

    # the genres are listed in a single column inside a long string. First, we unwrap this string, but appending the whole dataframe at the 
    # bottom of itself over and over, each time collecting a single genre in a new "myGenre" column
    df = movies[['primaryTitle','startYear','genres','averageRating','numVotes','runtimeMinutes']]
    genre_df = df['genres'].str.split(',', expand=True)
    max_nr_genres = len(genre_df.columns) # counts maximum number of genres listed per movie

    # do the appending after we extracted all individual genres as column
    genre_df[0].rename('myGenre', inplace=True)
    movies_genre_unfolded = pd.concat([df, genre_df[0]], axis = 1) # append genre as a column

    for k in range(1, max_nr_genres):
        genre_df[k].rename('myGenre', inplace=True)
        tmp = pd.concat([df, genre_df[k]], axis = 1) # append k'th genre as a column
        movies_genre_unfolded = pd.concat([movies_genre_unfolded, tmp], axis = 0) # append above dataframe at bottom of total frame
    movies_genre_unfolded.reset_index(drop=True, inplace=True)
    # clean
    movies_genre_unfolded.drop('genres', inplace=True, axis=1, errors='ignore')
    movies_genre_unfolded['myGenre'].fillna(value='no genre', inplace=True)
    # calculate the mean runtime and clean
    mean_runtime_genre_year = movies_genre_unfolded.groupby(by=['startYear','myGenre'])['runtimeMinutes'].mean().rename('meanRuntime', inplace=True)
    mean_runtime_genre_year = pd.DataFrame(mean_runtime_genre_year)
    mean_runtime_genre_year.reset_index(inplace=True)
    mean_runtime_genre_year = mean_runtime_genre_year[mean_runtime_genre_year['myGenre']!='no genre'] # drop no genre rows
    mean_runtime_genre_year['meanRuntime'] = mean_runtime_genre_year['meanRuntime'].round(2)

    return mean_runtime_genre_year

def wrangle_revenue_genre_year(movies):
    """Wrangle data for the line plot of avg. revenue by genre over the years."""

    # the genres are listed in a single column inside a long string. First, we unwrap this string, by appending the whole dataframe at the 
    # bottom of itself over and over, each time collecting a single genre in a new "myGenre" column
    df = movies[['primaryTitle','startYear','genres','averageRating','numVotes','runtimeMinutes','revenue']]
    genre_df = df['genres'].str.split(',', expand=True)
    max_nr_genres = len(genre_df.columns) # counts maximum number of genres listed per movie

    # do the appending after we extracted all individual genres as column
    genre_df[0].rename('myGenre', inplace=True)
    movies_genre_unfolded = pd.concat([df, genre_df[0]], axis = 1) # append genre as a column

    for k in range(1, max_nr_genres):
        genre_df[k].rename('myGenre', inplace=True)
        tmp = pd.concat([df, genre_df[k]], axis = 1) # append k'th genre as a column
        movies_genre_unfolded = pd.concat([movies_genre_unfolded, tmp], axis = 0) # append above dataframe at bottom of total frame
    movies_genre_unfolded.reset_index(drop=True, inplace=True)
    # clean
    movies_genre_unfolded.drop('genres', inplace=True, axis=1, errors='ignore')
    movies_genre_unfolded['myGenre'].fillna(value='no genre', inplace=True)
    # calculate the mean revenue and clean
    mean_revenue_genre_year = movies_genre_unfolded.groupby(by=['startYear','myGenre'])['revenue'].mean().rename('meanRevenue', inplace=True)
    mean_revenue_genre_year = pd.DataFrame(mean_revenue_genre_year)
    #mean_revenue_genre_year = mean_revenue_genre_year[mean_revenue_genre_year['myGenre']!='no genre'] # drop no genre rows
    mean_revenue_genre_year.reset_index(inplace=True)
    mean_revenue_genre_year = mean_revenue_genre_year.fillna(0)

    return mean_revenue_genre_year

def create_wordcloud(movies):
    """Prepare the string of movie titles to be used in a word cloud"""

    title_corpus = ' '.join(movies['primaryTitle'].unique())
    title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=1000, width=2000,
    colormap = sns.color_palette("plasma", as_cmap=True), random_state=1).generate(title_corpus)

    return title_wordcloud

def return_by_year(movies):
    # add a feature for monetary return
    df_return = movies.copy()
    df_return = df_return.dropna(subset=['revenue','budget'], axis=0)
    df_return['return']  = (df_return['revenue']-df_return['budget']) / df_return['budget'] # in percent
    df_return.reset_index(drop=True, inplace=True)

    min_budget = 50000 # set a minimum budget for movies to be considered here
    # select movies with most return
    max_return_by_year = df_return[(df_return['return'].notnull()) & (df_return['budget'] >= min_budget)][['primaryTitle', 'return', 'budget', 'revenue', 'startYear']]
    max_return_by_year = max_return_by_year.sort_values('return',ascending=False).drop_duplicates('startYear')
    #max_return_by_year.rename(columns={'primaryTitle':'Title','return':'N-fold return','budget':'Budget [$]','revenue':'Revenue [$]','startYear':'Year'}, inplace=True)
    max_return_by_year = max_return_by_year.round(2)
    max_return_by_year = max_return_by_year.sort_values('startYear',ascending=True)
    max_return_by_year.reset_index(drop=True, inplace=True)

    # select movies with worst return
    min_return_by_year = df_return[(df_return['return'].notnull()) & (df_return['budget'] >= min_budget)][['primaryTitle', 'return', 'budget', 'revenue', 'startYear']]
    min_return_by_year = min_return_by_year.sort_values('return',ascending=True).drop_duplicates('startYear')
    #max_return_by_year.rename(columns={'primaryTitle':'Title','return':'N-fold return','budget':'Budget [$]','revenue':'Revenue [$]','startYear':'Year'}, inplace=True)
    min_return_by_year = min_return_by_year.round(3)
    min_return_by_year = min_return_by_year.sort_values('startYear',ascending=True)
    min_return_by_year = min_return_by_year[min_return_by_year['return']<0] # remove positive returns
    min_return_by_year.reset_index(drop=True, inplace=True)

    return max_return_by_year, min_return_by_year

def dialogue_genre_scores(title_corpus):
    """Extract the neg. and pos. sentiment analysis compound scores for every genre."""

    # get unqiue genres list
    title_corpus.dropna(subset=['genresIMDB'], inplace=True, axis=0)
    title_corpus.reset_index(drop=True, inplace=True)
    genres = title_corpus['genresIMDB'].str.split(' ')
    res = []
    for k in range(len(genres)):
        for genre in genres[k]:
            res.append(genre)
    res = pd.Series(res)
    genre_list = res.unique()

    # find pos and neg score for each genre
    neg_score_ratings = []
    pos_score_ratings = []
    for genre in genre_list:
        val = title_corpus[title_corpus['genresIMDB'].str.contains(genre)].groupby('project_score').mean()['ratingIMDB']
        try:
            neg_score_ratings.append(val.iloc[0])
        except:
            neg_score_ratings.append(np.nan)
        try:
            pos_score_ratings.append(val.iloc[1])
        except:
            pos_score_ratings.append(np.nan)

    # find mean number of votes for each genre for pos and neg sentiment
    neg_votes_ratings = []
    pos_votes_ratings = []
    for genre in genre_list:
        val = title_corpus[title_corpus['genresIMDB'].str.contains(genre)].groupby('project_score').mean()['votes']
        try:
            neg_votes_ratings.append(val.iloc[0])
        except:
            neg_votes_ratings.append(np.nan)
        try:
            pos_votes_ratings.append(val.iloc[1])
        except:
            pos_votes_ratings.append(np.nan)

    # find count of movie sentiment pos or neg. per genre 
    neg_score_count = []
    pos_score_count = []
    for genre in genre_list:
        val = title_corpus[title_corpus['genresIMDB'].str.contains(genre)].groupby('project_score').count()['title']
        try:
            neg_score_count.append(val.iloc[0])
        except:
            neg_score_count.append(np.nan)
        try:
            pos_score_count.append(val.iloc[1])
        except:
            pos_score_count.append(np.nan)


    return genre_list, pos_score_ratings, neg_score_ratings, pos_score_count, neg_score_count, pos_votes_ratings, neg_votes_ratings

def keyword_genre_scores(genome_movies_sentiment):
    """Extract the neg. and pos. sentiment analysis compound scores for every genre based on keywords (not dialogue)"""
    
    title_corpus = genome_movies_sentiment.copy()
    # get unqiue genres list
    genres = title_corpus['genres'].str.split(',')
    res = []
    for k in range(len(genres)):
        for genre in genres[k]:
            res.append(genre)
    res = pd.Series(res)
    genre_list = res.unique()
    genre_list = genre_list[0:-4] # remove some useless genres

    # find pos and neg mean rating for each genre
    neg_score_ratings = []
    pos_score_ratings = []
    for genre in genre_list:
        val = title_corpus[title_corpus['genres'].str.contains(genre)].groupby('project_score').mean()['averageRating']
        try:
            neg_score_ratings.append(val.iloc[0])
        except:
            neg_score_ratings.append(np.nan)
        try:
            pos_score_ratings.append(val.iloc[1])
        except:
            pos_score_ratings.append(np.nan)

    # find mean number of votes for each genre for pos and neg sentiment
    neg_votes_ratings = []
    pos_votes_ratings = []
    for genre in genre_list:
        val = title_corpus[title_corpus['genres'].str.contains(genre)].groupby('project_score').mean()['numVotes']
        try:
            neg_votes_ratings.append(val.iloc[0])
        except:
            neg_votes_ratings.append(np.nan)
        try:
            pos_votes_ratings.append(val.iloc[1])
        except:
            pos_votes_ratings.append(np.nan)

    # find count of movie sentiment pos or neg. per genre 
    neg_score_count = []
    pos_score_count = []
    for genre in genre_list:
        val = title_corpus[title_corpus['genres'].str.contains(genre)].groupby('project_score').count()['primaryTitle']
        try:
            neg_score_count.append(val.iloc[0])
        except:
            neg_score_count.append(np.nan)
        try:
            pos_score_count.append(val.iloc[1])
        except:
            pos_score_count.append(np.nan)

    # mean sentiment by year
    mean_score_year = title_corpus.groupby('startYear').mean()['sentiment_score']

    return genre_list, pos_score_ratings, neg_score_ratings, pos_score_count, neg_score_count, pos_votes_ratings, neg_votes_ratings, mean_score_year


