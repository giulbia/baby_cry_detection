# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


__all__ = [
    'TrainClassifier'
]


class TrainClassifier:
    """
    Class to train a classifier of audio signals
    """

    def __init__(self):
        pass

    def train(self, dataset):
        """
        Train Random Forest

        :param dataset: pandas DataFrame with all features and actual label of signals
        :return: pipeline, best_param, best_estimator, perf
        """

        # Isolate features and label
        X = dataset.iloc[:, :-1]
        y = dataset['label']

        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

        pipeline = Pipeline([
            ('scl', StandardScaler()),
            ('clf', SVC(probability=True))
        ])

        # GridSearch
        param_grid = [{'clf__kernel': ['linear'], 'clf__C': [1, 1.5, 2, 5]}]

        estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

        model = estimator.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        perf = {'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, average='micro'),
                'precision': precision_score(y_test, y_pred, average='micro'),
                'f1': f1_score(y_test, y_pred, average='micro')}

        return perf, model.best_params_, model.best_estimator_



