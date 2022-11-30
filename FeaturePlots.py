from Importing import *


def create_3D_plot(playlist_train):
    # generate data
    n = 200

    # axes instance
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=-100)
    fig.add_axes(ax)

    # plot
    sc = ax.scatter(playlist_train['duration_ms'], playlist_train['tempo'], playlist_train['loudness'],
                    s=40, c=playlist_train['playlist_uri'], marker='o', cmap='flag')
    ax.set_xlabel('Duration (in ms)')
    ax.set_ylabel('Tempo')
    ax.set_zlabel('Loudness')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show();


class FeaturePlots:
    def __init__(self, path, files, x=5, y=0, plot3D=False):
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
        for file in files:
            self.file_name = file.split('.')[0]
            self.df = pd.read_csv(self.path + '/' + file)
            self.df['row_num'] = np.arange(len(self.df))
            self.create_plot()

    def create_plot(self):
        try:
            os.mkdir(f'feature_plots/{self.file_name}')
        except FileExistsError as error:
            print(f"FileExistsError::{error}")
        finally:
            x1 = self.df.iloc[:, self.x]
            x2 = self.df.iloc[:, self.y]
            sns.scatterplot(x=x1, y=x2, hue=self.df['playlist_uri'], palette='bright')
            plt.savefig(f"feature_plots/{self.file_name}/{self.df.keys()[self.y]}.png", dpi=150, facecolor='w')
            plt.show();
            if self.plot3D:
                playlist_train = pd.read_csv(f"{self.path}/{self.file_name}.csv")
                create_3D_plot(playlist_train)


path = 'extracted_data/csv'
files = os.listdir(path)

for i in range(0, 4):
    FeaturePlots(path, files, 5, i)

FeaturePlots(path, files, 5, 0)

