import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle


def learn(X, y, filename):
    clf = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,
        beta_2=0.999, early_stopping=False, epsilon=1e-08,
        hidden_layer_sizes=(3*256, 20), learning_rate='constant',
        learning_rate_init=0.001, max_iter=200, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True,
        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
        warm_start=False)

    clf.fit(X, y)
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)
        return True

    return False

def classify(X, filename):
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
        results = clf.predict_proba(X)
        return results
    return -1

    
if __name__ == "__main__":
    print("Testy:")

    X = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    y = [0, 1, 1, 1, 1, 0, 0]

    learn(X, y, "tests")

    print(classify([[0, 0, 1]], "tests"))
    print(classify([[0, 1, 1]], "tests"))