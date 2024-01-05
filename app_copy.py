
from sklearn.base import defaultdict
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
import plotly.express as px
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import openpyxl


CLIENT_ID = "dec5d6c9beb3410eb88c5648a53015ed"
CLIENT_SECRET = "99587f4565fb47799b9ad5e790ba75f4"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#Function to get albulm cover
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"





# Load your data
data = pd.read_excel("Spotify_1986_2023.xlsx")



# Drop unecessary column
data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

# Removes all duplicates
data = data.drop_duplicates()  

#Drop the rows with missing values 
data = data.dropna()





# List of numerical columns to consider for similarity calculations
# Use several numerical columns to consider for similarity calculations
Num_columns = ['popularity','acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'duration_ms',
        'principal_artist_followers', 'duration_min']



# Function to retrieve song data for a given song name
def find_song_data(song_name, dataset):
    try:
        return dataset[dataset['track_name'].str.lower() == song_name].iloc[0]
    except IndexError:
        return None



# Function to calculate the mean vector of a list of songs
def calculate_mean_vector(song_list, dataset):
    song_vectors = []
    
    for song_info in song_list:
        song_data = find_song_data(song_info['track_name'], dataset)
        
        if song_data is None:
            print(f"sorry: {song_info['track_name']} is not found in the list") 
            return None
        
        song_vector = song_data[Num_columns].values
        song_vectors.append(song_vector)
    
    if not song_vectors:
        print("Warning: No valid songs in the list")
        return None
    
    song_matrix = np.array(song_vectors)
    mean_vector = np.mean(song_matrix, axis=0)
    
    return mean_vector


# Function to flatten a list of dictionaries into a single dictionary
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


# Normalize the song data using Min-Max Scaler
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data[Num_columns])

# Standardize the normalized data using Standard Scaler
standard_scaler = StandardScaler()
scaled_normalized_data = standard_scaler.fit_transform(normalized_data)



# Function to recommend songs based on a list of seed songs
def recommend_songs(seed_songs, data, n_recommendations=10):
    metadata_cols = ['track_name', 'principal_artist_name', 'year', 'popularity']
    song_center = calculate_mean_vector(seed_songs, data)

    # Return an empty list if song_center is missing
    if song_center is None:
        return []

    # Normalize the song center
    normalized_song_center = min_max_scaler.transform([song_center])

    # Standardize the normalized song center
    scaled_normalized_song_center = standard_scaler.transform(normalized_song_center)

    # Calculate Euclidean distances and get recommendations
    distances = cdist(scaled_normalized_song_center, scaled_normalized_data, 'euclidean')
    index = np.argsort(distances)[0]

    # Filter out seed songs and duplicates, then get the top n_recommendations
    rec_songs = []
    seen_track_names = set()  # Keep track of seen track names to avoid duplicates
    for i in index:
        song_data = data.iloc[i]
        track_name = song_data['track_name'].lower()
        if track_name not in [song['track_name'] for song in seed_songs] and track_name not in seen_track_names:
            album_cover_url = get_song_album_cover_url(song_data['track_name'], song_data['principal_artist_name'])
            rec_songs.append({**song_data[metadata_cols].to_dict(), 'album_cover_url': album_cover_url})
            seen_track_names.add(track_name)
            if len(rec_songs) == n_recommendations:
                break

    return rec_songs



# set a good looking font
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Larger title using markdown
st.markdown('# Music Recommendation System' )

# Display text with larger font size
st.markdown(""" 
    Welcome to our Content-Based Music Recommendation System!
    
    This system helps you discover songs similar to the ones you like. 
    It considers features like the popularity of the songs (metadata features) and various sound aspects (audio features).
    The recommended songs will be listed in order, with the most similar songs appearing first and followed by the less similar ones.
    Our collection includes music from the years 1986 to 2023, ensuring a diverse selection for your musical journey.
    If you don't know which song to input, check the top 10 most popular songs in 2023.
    You can input more than one song.
    
    We hope you enjoy the experience!
""")



# Input for song names (use st.text_input or st.text_area)
song_names = st.text_area("#### Enter your song names (one per line):")

# Slider to select the number of recommendations
# The minimum value for the slider is set to 1.
# The maximum value for the slider is set to 30.
# The initial/default value is set to 15.
#n_recommendations = st.slider("Select the number of recommendations that you want:", 1, 30, 15)
n_recommendations = 15

# Convert input to list of song names
input_song_names = song_names.strip().split('\n') if song_names else []

# Button to recommend songs
# Button to recommend songs
if st.button('Recommend'):
    # Convert input to list of seed songs
    seed_songs = [{'track_name': name.lower()} for name in input_song_names]

    # Filter out empty names
    seed_songs = [song for song in seed_songs if song['track_name']]

    # If user does not enter input
    if not seed_songs:
        st.warning("Please enter at least one song name.")
    else:
        # Call the recommend_songs function
        recommended_songs = recommend_songs(seed_songs, data, n_recommendations)

        if not recommended_songs:
            st.warning("The provided songs are not in the list.")
        else:
            st.markdown('#### The Top ' + str(n_recommendations) + ' Similar Songs')
            # Display recommended songs
            for i, song in enumerate(recommended_songs):
                st.markdown(f"**{i + 1}. {song['track_name']}** by {song['principal_artist_name']} ({song['year']})")
                st.image(song['album_cover_url'], width=170)  # Adjust the width as needed
                


st.markdown("***")


#After this code-tiada masalah dh

# Display the top songs by popularity with album covers for the top 5 in 2023
st.subheader('Top 10 Most Popular songs in 2023')

# Convert 'year' column to datetime format
data['year'] = pd.to_datetime(data['year'], format='%Y')

# Get the top 5 most popular songs in 2023
top_10_songs_data = data[data['year'].dt.year == 2023].nlargest(10, 'popularity')

# Create two rows with 5 columns each
col1, col2, col3, col4, col5 = st.columns(5)
col6, col7, col8, col9, col10 = st.columns(5)

# Display the first 5 songs in the first row
with col1:
    st.text(top_10_songs_data.iloc[0]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[0]['track_name'], top_10_songs_data.iloc[0]['principal_artist_name']), width=130)
with col2:
    st.text(top_10_songs_data.iloc[1]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[1]['track_name'], top_10_songs_data.iloc[1]['principal_artist_name']), width=130)
with col3:
    st.text(top_10_songs_data.iloc[2]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[2]['track_name'], top_10_songs_data.iloc[2]['principal_artist_name']), width=130)
with col4:
    st.text(top_10_songs_data.iloc[3]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[3]['track_name'], top_10_songs_data.iloc[3]['principal_artist_name']), width=130)
with col5:
    st.text(top_10_songs_data.iloc[4]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[4]['track_name'], top_10_songs_data.iloc[4]['principal_artist_name']), width=130)

# Display the next 5 songs in the second row
with col6:
    st.text(top_10_songs_data.iloc[5]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[5]['track_name'], top_10_songs_data.iloc[5]['principal_artist_name']), width=130)
with col7:
    st.text(top_10_songs_data.iloc[6]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[6]['track_name'], top_10_songs_data.iloc[6]['principal_artist_name']), width=130)
with col8:
    st.text(top_10_songs_data.iloc[7]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[7]['track_name'], top_10_songs_data.iloc[7]['principal_artist_name']), width=130)
with col9:
    st.text(top_10_songs_data.iloc[8]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[8]['track_name'], top_10_songs_data.iloc[8]['principal_artist_name']), width=130)
with col10:
    st.text(top_10_songs_data.iloc[9]['track_name'])
    st.image(get_song_album_cover_url(top_10_songs_data.iloc[9]['track_name'], top_10_songs_data.iloc[9]['principal_artist_name']), width=130)







st.markdown("***")

#Music are often analyzed in terms of decades, so I will add a decade column to the main dataset.


# Convert year to datetime and extract decade
data['year'] = pd.to_datetime(data['year'], format='%Y')

# add a column of dacade to the dataset 
data['decade'] = (data['year'].dt.year - 1) // 10 * 10


# Count the number of songs per decade
decade_counts = data['decade'].value_counts().sort_index()

# Display the number of songs per decade
st.subheader('Number of Songs release per decade')
fig_decades = px.bar(x=decade_counts.index, y=decade_counts.values,
                     labels={'x': 'Decade', 'y': 'Number of Songs'},
                     title='Number of Songs per Decade', color=decade_counts.values)
fig_decades.update_layout(xaxis_type='category', height=1000, width=1000)
st.plotly_chart(fig_decades)


st.markdown("***")



# Display the distribution of song attributes using a histogram
st.subheader('Distribution of all Song Features')
st.markdown("These are all the song features that have been used to find your similar songs")
attribute_to_plot = st.selectbox('Select a feature to plot:', Num_columns)
fig_histogram = px.histogram(data, x=attribute_to_plot, nbins=30,
                              title=f'Distribution of {attribute_to_plot}')
fig_histogram.update_layout(height=1000, width=1000)
st.plotly_chart(fig_histogram)



st.markdown("***")


# Display a bar plot of artists with the most songs in the dataset
st.subheader('Artists with the Highest Song Counts')
top_artists = data['principal_artist_name'].str.replace("[", "").str.replace("]", "").str.replace("'", "").value_counts().head(20)
fig_top_artists = px.bar(top_artists, x=top_artists.index, y=top_artists.values, color=top_artists.index,
                         labels={'x': 'Artist', 'y': 'Number of Songs'},
                         title='Top Artists with Most Songs')
fig_top_artists.update_xaxes(categoryorder='total descending')
fig_top_artists.update_layout(height=1000, width=1000, showlegend=False)
st.plotly_chart(fig_top_artists)






