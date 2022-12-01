import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests

"""

80BPM = 3dy93B7qOpbohJGA85GlpE
140BPM = 34sYTZpdijvssbF0wlgmjr

1xAkAFbB3lUOpI9VpX2kbo

D-Minor = 6EhYrabdH4NmPblTHPcmty
C-Major = 7ctrM0MYkqbRLPnJRAYwwt

High tempo = 0uKDBOqgJI0amouaT5JHtN
Low Tempo = 1xwdBUUBvT3Dp3IRov3EjB

Adrenaline = 37i9dQZF1DXe6bgV3TmZOL
Jazz = 37i9dQZF1DX52ln8eMkne9

Meme = 7hmKXGvxszjvAynIlRZglO
Ghazal = 37i9dQZF1DXdsiL9gD4vAA

00s Rock Anthem : 37i9dQZF1DX3oM43CtKnRV
Rock This : 37i9dQZF1DXcF6B6QPhFDv
Chill Vibes : 37i9dQZF1DX889U0CL85jj
"""


CID = "42225bb4f6fe41ac9a3c7f5342e27f91"
SECRET_KEY = "ba991ae9221e41a581fa96b5c827d6d0"
AUTH_URL = 'https://accounts.spotify.com/api/token'
BASE_URL = 'https://api.spotify.com/v1/'
FILE_NAME = "track_features_Old_Rock_&_New_Rock"
PLAYLISTS = ['37i9dQZF1DX3oM43CtKnRV', '37i9dQZF1DXcF6B6QPhFDv']


class PlaylistFeatureExtraction:
    def __init__(self):
        self.ply_uri = []
        self.trk_uri = []
        self.trk_num = []
        self.trk_nm = []
        self.audio_features = []
        self.audio_analysis = []
        self.AUTH_RESPONSE = None
        self.sp = None
        self.send_request()
        self.fetch_playlist_data()

    def send_request(self):
        self.AUTH_RESPONSE = requests.post(AUTH_URL, {
            'grant_type': 'client_credentials',
            'client_id': CID,
            'client_secret': SECRET_KEY,
        })

        auth_response_data = self.AUTH_RESPONSE.json()
        access_token = auth_response_data['access_token']
        headers = {
            'Authorization': 'Bearer {token}'.format(token=access_token)
        }
        client_credentials_manager = SpotifyClientCredentials(client_id=CID, client_secret=SECRET_KEY)
        self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def fetch_playlist_data(self):
        i = 0
        for pl in PLAYLISTS:

            print(f"{i}-{len(PLAYLISTS)-1}:PLAYLIST:START")
            track_uris = [x["track"]["uri"] for x in self.sp.playlist_tracks(pl)["items"]]
            self.trk_uri.append(track_uris)

            for trk in track_uris:
                self.ply_uri.append(pl)
                self.trk_num.append(trk)
                self.trk_nm.append(self.sp.track(trk)["name"])
                self.audio_features.append(self.sp.audio_features(trk))
                self.audio_analysis.append(self.sp.audio_analysis(trk))
            print(f"{i}-{len(PLAYLISTS)-1}:PLAYLIST:END")
            i = i + 1

        trk_fet = pd.DataFrame(
            {
                'Playlist URI': self.ply_uri,
                'Track URI': self.trk_num,
                'Track Name': self.trk_nm,
                'Audio Features': self.audio_features,
                "Audio Analysis": self.audio_analysis})

        trk_fet.to_excel(f"extracted_data/excel/{FILE_NAME}.xlsx")


PlaylistFeatureExtraction()
