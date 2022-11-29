#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:03:14 2022

@author: abhik_bhattacharjee
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests


ply_uri = []
trk_uri = []
trk_num = []
trk_nm = []
audio_features = []
audio_analysis = []

playlist_list = ['37i9dQZF1DX5cZuAHLNjGz', '37i9dQZF1DWWQRwui0ExPn', 
                 '37i9dQZF1DXcF6B6QPhFDv', '37i9dQZF1DX0DxcHtn4Hwo']

cid = "42225bb4f6fe41ac9a3c7f5342e27f91"
sec_key = "ba991ae9221e41a581fa96b5c827d6d0"

AUTH_URL = 'https://accounts.spotify.com/api/token'

auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': cid,
    'client_secret': sec_key,
})

auth_response_data = auth_response.json()

access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}
BASE_URL = 'https://api.spotify.com/v1/'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=sec_key)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

for pl in playlist_list:    
    #print(pl)
    track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(pl)["items"]]
    #print(track_uris)
    trk_uri.append(track_uris)

    for trk in track_uris:
        ply_uri.append(pl)
        trk_num.append(trk)
        trk_nm.append(sp.track(trk)["name"])
        audio_features.append(sp.audio_features(trk))
        audio_analysis.append(sp.audio_analysis(trk))
    
trk_fet = pd.DataFrame(
    {
     'Playlist URI' : ply_uri,
     'Track URI' : trk_num, 
     'Track Name' : trk_nm,
     'Audio Features' : audio_features,
     "Audio Analysis" : audio_analysis})
trk_fet.to_excel("track_features.xlsx")