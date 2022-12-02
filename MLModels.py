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
    def __init__(self, path=csv_path, file=csv_file, create_poly=False, normalization=True, kf_splits=5):
        (self.X_train, self.X_test), (self.y_train, self.y_test) = (None, None), (None, None)
        self.originalX, self.originalY = None, None
        self.kf_splits = kf_splits
        self.kf = KFold(n_splits=kf_splits)
        self.mse_data = []
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
        self.originalX = np.array(X)
        self.originalY = np.array(y)
        if self.create_poly:
            polyModel = PolynomialFeatures(degree=DEGREE)
            X = polyModel.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        return (self.X_train, self.X_test), (self.y_train, self.y_test)

    def train_data(self):
        for train, test in self.kf.split(self.originalX):
            self.model.fit(self.originalX[train], self.originalY[train])
            y_pred_temp = self.model.predict(self.originalX[test])
            mse = mean_squared_error(self.originalY[test], y_pred_temp)
            print("MEAN SQUARED ERROR: {}".format(mse))
            self.mse_data.append(mse)
        print(self.model)
        self.model.fit(self.X_train, self.y_train)

    def get_kfold_data(self):
        return np.array(self.mse_data).mean(), np.array(self.mse_data).std()

    def print_metrics(self):
        y_pred = self.model.predict(self.X_test)
        metrics(self.y_test, y_pred)


class KNeighborsModel(TrainingData):
    def __init__(self, n_neighbours=5):
        super().__init__()
        self.model = KNeighborsClassifier(metric='cosine', n_neighbors=n_neighbours)
        self.train_data()
        self.print_metrics()


class LogisticRegressionModel(TrainingData):
    def __init__(self, create_poly=False, C=1.0):
        super().__init__(create_poly=create_poly)
<<<<<<< HEAD
        self.model = LogisticRegression(C=C)
=======
        self.model = LogisticRegression(penalty='l2')
>>>>>>> parent of 67210f5 (Few Changes in the code)
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

def calculate_cross_val(model, X, y,mean_error, std_error, scoring_pattern='f1'):
    score = cross_val_score(model, X, y, cv=5, scoring=scoring_pattern)
    mean_error.append(np.array(score).mean())
    std_error.append(np.array(score).std())
    return mean_error, std_error

<<<<<<< HEAD

K_vals = [3, 5, 7, 9, 11]
mean_arr, std_arr = [], []
for k in K_vals:
    KNN = KNeighborsModel(n_neighbours=k)
    mean, std = KNN.get_kfold_data()
    #mean_arr, std_arr = calculate_cross_val(KNN.model, KNN.X_test, KNN.y_test, mean_arr, std_arr)
    mean_arr.append(mean)
    std_arr.append(std)
### Error bar
fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(K_vals, mean_arr, yerr=std_arr)
plt.xlabel('K')
plt.ylabel('Mean square error')
plt.xlim((np.min(K_vals) - 0.1, np.max(K_vals) + 0.1))
plt.title("Errorbar for KNN")
#plt.show()
plt.savefig("metric_plots/KNeighborsClassifier/track_features_Rock_LoFi_Country/error_bar.png")


"""
    Logistic regression
"""
C_Values = [1, 10, 20, 30, 40, 50]
mean_arr, std_arr = [], []
for c in C_Values:
    Logi = LogisticRegressionModel(create_poly=True,C=c)
    mean, std = Logi.get_kfold_data()
    #mean_arr,std_arr = calculate_cross_val(Logi.model, Logi.X_test, Logi.y_test, mean_arr, std_arr)
    mean_arr.append(mean)
    std_arr.append(std)
### Error bar
fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(C_Values, mean_arr, yerr=std_arr)
plt.xlabel('C')
plt.ylabel('Mean square error')
plt.xlim((np.min(C_Values) - 0.1, np.max(C_Values) + 0.1))
plt.title("Errorbar for Logistic Regression\n(degree = 2)")
# plt.show()
plt.savefig("metric_plots/LogisticRegression/track_features_Rock_LoFi_Country/error_bar.png")
# MLP = MLPModel(hidden_layer=(200, 150, 100, 50))
# NaiveBayes = NaiveBayesModel()
=======
KNN = KNeighborsModel()
Logi = LogisticRegressionModel(create_poly=True)
SVC = SVCModel(create_poly=True)
MLP = MLPModel(hidden_layer=(200, 150, 100, 50))
NaiveBayes = NaiveBayesModel()
MultinomialNaiveBayes = MultinomialNaiveBayesModel()
>>>>>>> parent of 67210f5 (Few Changes in the code)
