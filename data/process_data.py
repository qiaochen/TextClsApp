#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:44:11 2018
@author: chen
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
TABLE_NAME = "tb_msg"

def load_data(messages_filepath, categories_filepath):
    """
    load data from raw text files
    :param messages_filepath: path for the message file
    :param categories_filepath: path for the categories file
    :return: merged dataframe
    """
    df_msg = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    return df_msg.merge(df_cat, on='id')


def clean_data(df):
    """
    Clean and transform data
    :param df: merged dataframe
    :return: cleaned dataframe
    """
    categories = pd.DataFrame(df.categories.apply(lambda x: x.split(';')).tolist())
    row = categories.iloc[0, :]
    category_colnames = [ele[:ele.index('-')] for ele in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[x.index('-') + 1:])
        categories[column] = categories[column].astype('int')
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop(df.index[df.id.duplicated()], axis=0)
    return df


def save_data(df, database_filename):
    """
    Save df to sqlite database
    :param df: dataframe to be persisenced
    :param database_filename:  the export database name
    :return: saved path name
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(TABLE_NAME, engine, index=False)
    return database_filename


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()