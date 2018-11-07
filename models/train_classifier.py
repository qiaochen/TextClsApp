#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:44:11 2018
@author: chen
"""


from sqlalchemy import create_engine
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import sys
from feature_extractor import TextStatisticsComputer, tokenize


ONE_LABEL_COL = "child_alone"
TABLE_NAME = "tb_msg"

def load_data(database_filepath):
    """
    Load data from sqlite databse
    :param database_filepath: database path
    :return: Feature and categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(TABLE_NAME,con=engine)
    X = df.message
    df_dropped = df.drop(ONE_LABEL_COL, axis=1) # the child_alone category has no positive cases
    Y = df_dropped.iloc[:,4:]
    return X, Y, list(df_dropped.columns[4:])


def build_model():
    """
    Build the model pipline
    :return: The pipline of the whole model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf_line', Pipeline([
                ('tfidf', TfidfVectorizer(analyzer=tokenize, min_df=3)),
                ('svd', TruncatedSVD(n_components=256)),
            ])),
            ('stats_line', Pipeline([
                ('statistics', TextStatisticsComputer()),
                ('std', StandardScaler()),
            ]))
        ])),
        ('clr', MultiOutputClassifier(GradientBoostingClassifier(
            loss="deviance"), n_jobs=-1))])
    parameters = {"clr__estimator__learning_rate": [0.1],
                  "clr__estimator__n_estimators": [100, 300],
                  'clr__estimator__max_depth': [1, 3],
                  'clr__estimator__max_leaf_nodes': [3, 5],
                  'clr__estimator__max_features': ['sqrt', 'log2']}
#    balanced_scorer = make_scorer(balanced_accuracy_score)
    gridcv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)
    print("Best score: {} \nBest params: {}".format(gridcv.best_score_, gridcv.best_params_))
    return gridcv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates trained model
    :param model: Trained model
    :param X_test: test features
    :param Y_test: test labels
    :param category_names: names of the labels
    :return: reports and f1 scores
    """
    reports = []
    f1s = []
    preds = model.predict(X_test)
    for idx, col in enumerate(category_names):
        reports.append(classification_report(Y_test.iloc[:, idx], preds[:, idx], output_dict =True))
        print("="*20)
        print(col)
        print(reports[-1])
        print("-"*20)
        f1s.append(reports[-1]['1']['f1-score'])
    print("Average F1 score: ", np.mean(f1s))
    return reports, f1s


def save_model(model, model_filepath):
    """
    Save the model
    :param model: trained model
    :param model_filepath: output path
    :return: output_path
    """
    with open(model_filepath, 'wb') as out_file:
        pickle.dump(model.best_estimator_, out_file)
    return model_filepath


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
