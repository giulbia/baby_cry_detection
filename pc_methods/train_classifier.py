# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import numpy as np
import logging
import timeit


__all__ = [
    'TrainClassifier'
]


class TrainClassifier:
    """
    Class to train a classifier of audio signals
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        """
        Train Random Forest

        :return: pipeline, best_param, best_estimator, perf
        """

        logging.info('Splitting train and test set. Test set size: 0.25%')

        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.25,
                                                            random_state=0,
                                                            stratify=self.y)

        logging.info('Train set size: {0}. Test set size: {1}'.format(y_train.size, y_test.size))

        pipeline = Pipeline([
            ('scl', StandardScaler()),
            # ('lda', LinearDiscriminantAnalysis()),
            ('clf', SVC(probability=True))
        ])

        # GridSearch
        param_grid = [{'clf__kernel': ['linear', 'rbf'],
                       'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                       'clf__gamma': np.logspace(-2, 2, 5),
                       # 'lda__n_components': range(2, 17)
                       }]

        estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

        logging.info('Training model...')
        start = timeit.default_timer()

        model = estimator.fit(X_train, y_train)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        y_pred = model.predict(X_test)

        perf = {'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'f1': f1_score(y_test, y_pred, average='macro'),
                # 'summary': classification_report(y_test, y_pred)
                }

        logging.info(perf)

        return perf, model.best_params_, model.best_estimator_



