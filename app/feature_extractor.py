#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:44:11 2018
@author: chen
"""

from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk
import pandas as pd

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class TextStatisticsComputer(BaseEstimator, TransformerMixin):
    """
    Compute text statistics
    """
    def __init__(self):
        """
        Initialize extractor
        """
        self.extractors = [self.text_len, self.n_url, self.n_ner, self.score_sentiment]

    def text_len(self, text):
        """
        compute text length
        :param text: message text
        :return: length of the text
        """
        return len(text)

    def n_url(self, text):
        """
        Compute number of urls in the text
        :param text: message text
        :return: the number of urls
        """
        detected_urls = re.findall(url_regex, text)
        return len(detected_urls)

    def n_ner(self, text):
        """
        Compute the number of named entities in the text
        :param text: message text
        :return: number of named entities
        """
        return len(self._get_continuous_chunks(text))

    def score_sentiment(self, text):
        """
        Compute the sentiment scores of the text
        :param text: message text
        :return: the float sentiment score [-1, +1]
        """
        blob = TextBlob(text)
        return blob.sentiment[0]

    def _get_continuous_chunks(self, text):
        """
        Compute continuous chunks of named entities
        :param text: message text
        :return: list of named entities
        """
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "@url")
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        continuous_chunk = []
        current_chunk = []

        for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        if continuous_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)

        return continuous_chunk

    def fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X):
        """
        Compute the statisitcal features for the dataset
        :param X:Raw text messages
        :return: the feature dataframe
        """
        X_statisitcs = pd.concat([pd.Series(X).apply(extractor) for extractor in self.extractors], axis=1)
        return pd.DataFrame(X_statisitcs)
    
def tokenize(text):
    """
    clean and tokenize text to words
    :param text: raw message text
    :return: cleaned and tokenized text
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "@url")
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words]
    return words
