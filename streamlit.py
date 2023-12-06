import streamlit as st
import pandas as pd
from io import StringIO


# two pages
    # 1. get asanas from video
        # 1. youtube
        # 2. drag/search file
        # 3. webcam
            # genera 3 liste: video_frame, guessed_asana, image_guessed_asana
            # possibilità di scorrere (prev-next asana)
            # bottone 'scopri di più' per avere informazioni sull'asana
    # 2. get asana information


st.set_page_config(page_title="AsanaNet", page_icon=":woman-cartwheeling:", layout="wide")


# 1.2
'''
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
'''