from Importing import *


def z_score(x):
    return (x - x.mean()) / (x.std())


def gaussian(x):
    return np.exp(-pow(x, 2))


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def metrics(y_test, y_pred):
    print("-"*10+"CONFUSION-MATRIX"+"-"*10)
    print(confusion_matrix(y_test, y_pred))
    print("-"*10+"CLASSIFICATION-REPORT"+"-"*10)
    print(classification_report(y_test, y_pred))


FEATURES = ['duration_ms','key', 'loudness', 'tempo']
NORMALIZATION = z_score
DEGREE = 2
path = 'extracted_data/csv'
files = os.listdir(path)


class TrainingData:
    def __init__(self, path=path, files=files, create_poly=False, normalization=True):
        (self.X_train, self.X_test), (self.y_train, self.y_test) = (None, None), (None, None)
        self.path = path
        self.files = files
        self.file_name = None
        self.playlist_train = None
        self.model = None
        self.create_poly = create_poly
        self.normalization = normalization
        self.read_file()
        if normalization:
            self.normalize_data()
        self.return_data()

    def read_file(self):
        for file in self.files:
            self.file_name = file.split('.')[0]
            self.playlist_train = pd.read_csv(f"{self.path}/{self.file_name}.csv")

    def normalize_data(self):
        for f in FEATURES:
            self.playlist_train[f] = NORMALIZATION(self.playlist_train[f])

    def return_data(self):
        X = self.playlist_train[FEATURES]
        y = self.playlist_train['playlist_uri']
        if self.create_poly:
            polyModel = PolynomialFeatures(degree=DEGREE)
            X = polyModel.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        return (self.X_train, self.X_test), (self.y_train, self.y_test)

    def train_data(self):
        print(self.model)
        self.model.fit(self.X_train, self.y_train)

    def print_metrics(self):
        y_pred = self.model.predict(self.X_test)
        metrics(self.y_test, y_pred)


class KNeighborsModel(TrainingData):
    def __init__(self):
        super().__init__()
        self.model = KNeighborsClassifier(metric='cosine')
        self.train_data()
        self.print_metrics()


class LogisticRegressionModel(TrainingData):
    def __init__(self, create_poly=False):
        super().__init__(create_poly=create_poly)
        self.model = LogisticRegression(penalty='l2')
        self.train_data()
        self.print_metrics()


class SVCModel(TrainingData):
    def __init__(self, create_poly=False):
        super().__init__(create_poly=create_poly)
        self.model = SVC(probability=True)
        self.train_data()
        self.print_metrics()


class MLPModel(TrainingData):
    def __init__(self, create_poly=False, hidden_layer=None):
        super().__init__(create_poly=create_poly)
        self.hidden_layer = hidden_layer
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer, max_iter=5,
                                   activation='relu', solver='adam', random_state=1)
        self.train_data()
        self.print_metrics()


class NaiveBayesModel(TrainingData):
    def __init__(self, create_poly=False, hidden_layer=None):
        super().__init__(create_poly=create_poly)
        self.hidden_layer = hidden_layer
        self.model = GaussianNB()
        self.train_data()
        self.print_metrics()


KNN = KNeighborsModel()
Logi = LogisticRegressionModel()
SVC = SVCModel(create_poly=True)
MLP = MLPModel(hidden_layer=(200, 150, 100, 50))
NaiveBayes = NaiveBayesModel()

print(NaiveBayes.model.score(NaiveBayes.X_test, NaiveBayes.y_test))
