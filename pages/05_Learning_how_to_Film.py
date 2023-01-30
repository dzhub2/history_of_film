# Imports
import streamlit as st
from  plotting_functions import *
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

### Define page-wide parameters
BGCOLOR = 'lightskyblue'
FONTSIZE = 17
FONTFAMILY = 'Garamond'
WIDTH_TO_CENTER = 3
PERSIST = False
CACHESUPPRESS = True

st.sidebar.markdown(":male-scientist: Learning how to Film :male-scientist:")

### Load film data
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_regression_result():
    return load_regression_result()

importances_return, importances_rating = st_load_regression_result()


### Start the Page
_, col2, _ = st.columns([0.1, 0.9, 0.1])
with col2:
    st.title(":male-scientist: Learning how to Film :male-scientist:")

_, col2, _ = st.columns([0.1, 0.2, 0.1])
with col2:
    st.video("https://www.youtube.com/watch?v=bBixD-rTB_c")

st.subheader("**Can Machine Learning be used to design the perfect film?**")
st.write("On this page, we attempt to find the most relevant features to produce a film with \
**maximum return** and **maximum IMDB rating**. To this end, **two regression models** were trained. Details on the implementation can be found in [this notebook]( !!!!!!!!!!!! LINK !!!!!!!!). \
")

st.markdown(
"""
The following features went into the models:
- Return on Investment (Target in Model 1)
- IMDB Rating (Target in Model 2)
- Budget
- Number of IMDB Votes
- Genres
- Film Runtime
- Number of Directors
- Number of Writers
- Native Language is English or not (Foreign)
"""
)

st.write("The best model performance was achieved using gradient boosted decision trees ([XGBoost](https://xgboost.ai/)).\
 The most important features in both models are shown below:")

## Plotting:
# styling
sns.color_palette("pastel")
sns.set(font="Garamond")
sns.set(rc={'axes.facecolor':'white'})
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.left'] = True
#sns.set_context(rc = {'patch.linewidth': 50})

# draw histogram of importances of each feature
# Return
thresh = 0.05

importances_return = importances_return[importances_return['Importance']>=thresh]
importances_return.sort_values(by='Importance', inplace=True, ascending=False)
plot_return = plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Feature', y='Importance', data=importances_return, edgecolor = "black", palette='magma')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Rating

importances_rating = importances_rating[importances_rating['Importance']>=thresh]
importances_rating.sort_values(by='Importance', inplace=True, ascending=False)
plot_rating = plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Feature', y='Importance', data=importances_rating, edgecolor = "black", palette='magma')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

st.header("Maximize Return on Investment")
st.pyplot(plot_return)
st.header("Maximize IMDB Rating")
st.pyplot(plot_rating)