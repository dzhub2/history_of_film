# Imports
import streamlit as st
from  streamlit_functions import *
import base64
import numpy as np

### Define page-wide parameters
BGCOLOR = 'lightskyblue'
FONTSIZE = 17
FONTFAMILY = 'Garamond'
WIDTH_TO_CENTER = 3
PERSIST = False
CACHESUPPRESS = True

st.sidebar.markdown(":question: Want to Play a Game :question:")

### Load film data
@st.cache(suppress_st_warning=CACHESUPPRESS, persist=PERSIST)
def st_load_quotes():
    return load_quotes()

quotes = st_load_quotes()
quotes = quotes # randomize the order
unique_movies = quotes['Movie Name'].sort_values().unique()

### Start the Page
_, col2, _ = st.columns([0.215, 2, 0.1])
with col2:
    st.title(":question: Can You Guess The Film :question:")

_, col2, _ = st.columns([0.1, 0.2, 0.1])
with col2:
    st.image("./play.gif")

# Initialization
if 'idx' not in st.session_state:
    st.session_state['idx'] = 1
if 'movies' not in st.session_state:
    st.session_state['quotes'] = quotes
if 'points' not in st.session_state:
    st.session_state['points'] = 0


with st.form("my_form", clear_on_submit=False):

    st.write(":notebook: Which film does this quote belong to :notebook::")
    col1, col2, _ = st.columns([9,5,9])
    with col2:
        st.write(f"{st.session_state['quotes']['Catchphrase'].iloc[st.session_state['idx']]}")


    user_movie = st.selectbox(label='Make your guess...', options=unique_movies)

    # Now add a submit button to the form:
    _, col2, _ = st.columns([9,3,9])
    with col2:
        submitted = st.form_submit_button("Submit")

    if submitted:

        if user_movie == st.session_state['quotes']['Movie Name'].iloc[st.session_state['idx']-1]:

            st.write(":tada: You're right :tada:! Have a point :100:.")
            st.session_state['points'] += 1 # add 1 to the total points

        else:
            st.write("That was wrong :smiling_face_with_tear:")

        if st.session_state['points'] == 1: # singular
            st.write(f"You've earned {st.session_state['points']} out of {st.session_state['idx']-1} points.")
        else: # plural
            st.write(f"You've earned {st.session_state['points']} out of {st.session_state['idx']-1} points.")

        st.write(f"**The right answer was:** {st.session_state['quotes']['Movie Name'].iloc[st.session_state['idx']-1]}")

        st.markdown("**The quote is used in the following context:**")
        st.write(f"{st.session_state['quotes']['Context'].iloc[st.session_state['idx']-1]}")

        
    st.session_state['idx'] += 1

_, col2, _ = st.columns([0.1, 0.25, 0.1])
with col2:
    st.video("https://www.youtube.com/watch?v=bBixD-rTB_c")