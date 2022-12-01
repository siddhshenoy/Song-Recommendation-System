from Importing import np


def z_score(x):
    return (x - x.mean()) / (x.std())


def gaussian(x):
    return np.exp(-pow(x, 2))


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

