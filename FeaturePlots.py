from Importing import *
from NormalizationFunctions import z_score, minmax, gaussian

FEATURES = ['duration_ms', 'key', 'loudness', 'tempo']
NORMALIZATION = z_score


def normalize_data(data):
    for f in FEATURES:
        data[f] = NORMALIZATION(data[f])


def create_3D_plot(playlist_train):
    # generate data
    n = 200

    # axes instance
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=-100)
    fig.add_axes(ax)

    for f in FEATURES:
        playlist_train[f] = NORMALIZATION(playlist_train[f])

    # plot
    sc = ax.scatter(playlist_train['duration_ms'], playlist_train['tempo'], playlist_train['loudness'],
                    s=40, c=playlist_train['playlist_uri'], marker='o', cmap='brg')
    ax.set_xlabel('Duration (in ms)')
    ax.set_ylabel('Tempo')
    ax.set_zlabel('Loudness')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show();


csv_path = 'extracted_data/csv'
csv_files = os.listdir(csv_path)


class FeaturePlots:
    def __init__(self, path=csv_path, files=csv_files, x=5, y=0, plot3D=False):
        self.path = path
        self.files = files
        self.x = x
        self.y = y
        self.plot3D = plot3D
        self.df = None
        self.file = None
        self.file_name = None
        self.process_data()

    def process_data(self):
        for file in self.files:
            self.file_name = file.split('.')[0]
            self.df = pd.read_csv(self.path + '/' + file)
            self.df['songs'] = np.arange(len(self.df))
            self.create_plot()

    def create_plot(self):
        try:
            os.mkdir(f'feature_plots/{self.file_name}')
        except FileExistsError as _:
            print(f"FOLDER:EXISTS")
        finally:
            self.normalize_data()
            x1 = self.df.iloc[:, self.x]
            x2 = self.df.iloc[:, self.y]

            sns.scatterplot(x=x1, y=x2, style=self.df['playlist_uri'], hue=self.df['playlist_uri'], palette='bright', markers=('o', 'P'))
            plt.title(" ".join([word.capitalize() for word in self.file_name.split('_')][:]))

            custom = [Line2D([], [], marker='o', color='b', linestyle='None'),
                      Line2D([], [], marker='P', color='orange', linestyle='None')]

            legend_values = []
            if "Adrenaline" in self.file_name:
                legend_values = ['Jazz', 'Adrenaline']

            if "Tempo" in self.file_name:
                legend_values = ['High Tempo', 'Low Tempo']

            if "Long" in self.file_name:
                legend_values = ['Long Songs', 'Short Songs']

            if "OldRock" in self.file_name:
                legend_values = ['Old Rock', 'New Rock', 'Chill Songs']

            plt.legend(custom, legend_values, loc='best', prop={'size': 13}, frameon=True)
            plt.savefig(f"feature_plots/{self.file_name}/{self.df.keys()[self.y]}.png", dpi=150, facecolor='w')
            plt.show()
            if self.plot3D:
                playlist_train = pd.read_csv(f"{self.path}/{self.file_name}.csv")
                create_3D_plot(playlist_train)

    def normalize_data(self):
        for f in FEATURES:
            self.df[f] = NORMALIZATION(self.df[f])



# for i in range(0, 4):
#     FeaturePlots(path, files, 5, i)
#
# FeaturePlots(path, files, 5, 0)

