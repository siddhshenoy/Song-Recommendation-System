from Importing import *
from NormalizationFunctions import z_score, minmax, gaussian


def metrics(y_test, y_pred):
    print("-"*10+"CONFUSION-MATRIX"+"-"*10)
    print(confusion_matrix(y_test, y_pred))
    print("-"*10+"CLASSIFICATION-REPORT"+"-"*10)
    print(classification_report(y_test, y_pred))


FEATURES = ['duration_ms', 'key', 'loudness', 'tempo']
NORMALIZATION = z_score
DEGREE = 2
csv_path = 'extracted_data/csv'
FILE_NAME = None


print(os.listdir(csv_path))
try:
    csv_file = os.listdir(csv_path)[int(input("Enter Choice: "))]
except IndexError as error:
    print("INDEX OUT OF RANGE")


class TrainingData:
    def __init__(self, path=csv_path, file=csv_file, create_poly=False, normalization=True):
        (self.X_train, self.X_test), (self.y_train, self.y_test) = (None, None), (None, None)
        self.path = path
        self.file = file
        self.file_name = self.file.split('.')[0]
        self.playlist_train = None
        self.model = None
        self.create_poly = create_poly
        self.normalization = normalization
        self.read_file()
        if self.normalization:
            self.normalize_data()
        self.return_data()

    def read_file(self):
        print('\n'+" ".join([word.lower() for word in self.file_name.split('_')])+'\n')
        self.playlist_train = pd.read_csv(f"{self.path}/{self.file}")

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


class MultinomialNaiveBayesModel(TrainingData):
    def __init__(self, create_poly=False, hidden_layer=None):
        super().__init__(create_poly=create_poly)
        self.hidden_layer = hidden_layer
        self.model = MultinomialNB()
        self.train_data()
        self.print_metrics()


KNN = KNeighborsModel()
Logi = LogisticRegressionModel(create_poly=True)
SVC = SVCModel(create_poly=True)
MLP = MLPModel(hidden_layer=(200, 150, 100, 50))
NaiveBayes = NaiveBayesModel()
MultinomialNaiveBayes = MultinomialNaiveBayesModel()