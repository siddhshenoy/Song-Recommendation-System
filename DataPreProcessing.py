from Importing import *

label_encoder = LabelEncoder()

FEATURES = ["playlist_uri", 'key', 'loudness', 'tempo', 'duration_ms']

FEATURE_STORE = {
    "key": [],
    "loudness": [],
    "tempo": [],
    "duration_ms": [],
    "playlist_uri": [],
}


class DataPreProcessing:
    def __init__(self, path, files):
        self.path = path
        self.files = files
        self.df = None
        self.file_name = None
        self.process_data()

    def process_data(self):
        for file in self.files:
            self.file_name = file.split('.')[0]
            self.df = pd.read_excel(self.path + '/' + file)
            self.df['Playlist URI'] = label_encoder.fit_transform(self.df['Playlist URI'])
            self.df['Playlist URI'].value_counts()

            temp_df = pd.DataFrame.from_dict(FEATURE_STORE)
            temp_df.to_csv(f'extracted_data/csv/{self.file_name}.csv', mode='w', index=False, header=True)

            self.populate_DataFrame()

    def populate_DataFrame(self):

        for audio_features, playlist in zip(self.df['Audio Features'], self.df['Playlist URI']):
            features = ast.literal_eval(audio_features)
            features[0]["playlist_uri"] = playlist
            for key in list(features[0].keys()):
                if key not in FEATURES:
                    del features[0][key]
                else:
                    features[0][key] = [features[0][key]]

            self.df = pd.DataFrame.from_dict(features[0])
            self.df.to_csv(f"extracted_data/csv/{self.file_name}.csv", mode='a', index=False, header=False)

        self.df = pd.read_csv(f'extracted_data/csv/{self.file_name}.csv')
        self.df = self.df.sample(frac=1)
        self.df.to_csv(f'extracted_data/csv/{self.file_name}.csv', mode='w', index=False, header=True)


path = 'extracted_data/excel'
files = os.listdir(path)
DataPreProcessing(path, files)
