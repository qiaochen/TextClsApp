import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
ONE_LABEL_COL = "child_alone"

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tb_msg', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    ## For genre analysis
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    ## Positive Categories
    pos_ratios = list(df[df.columns[4:]].mean(axis=1))
    cat_names = list(df.columns[4:])
    
    # create visuals
    p_graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=pos_ratios
                )
            ],

            'layout': {
                'title': 'Ratios of Positive Labels Per-Category',
                'yaxis': {
                    'title': "Ratio"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    p_ids = ["p_graph-{}".format(i) for i, _ in enumerate(p_graphs)]
    p_graphJSON = json.dumps(p_graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    ## Length hist gram
    # extract data needed for visuals
    message_lengths = df.message.apply(lambda text: len(text))
    
    # create visuals
    l_graphs = [
        {
            'data': [
                Histogram(x=message_lengths)
            ],

            'layout': {
                'title': 'Distribution of message lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        }
    ]
        
    # encode plotly graphs in JSON
    l_ids = ["l_graph-{}".format(i) for i, _ in enumerate(l_graphs)]
    l_graphJSON = json.dumps(l_graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    # Sentiment hist
    # extract data needed for visuals
    sentiments = df.message.apply(lambda text: TextBlob(text).sentiment[0])
    
    # create visuals
    s_graphs = [
        {
            'data': [
                Histogram(x=sentiments)
            ],

            'layout': {
                'title': 'Distribution of message sentiments [-1, +1]',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Sentiment score"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    s_ids = ["s_graph-{}".format(i) for i, _ in enumerate(s_graphs)]
    s_graphJSON = json.dumps(s_graphs, cls=plotly.utils.PlotlyJSONEncoder)

    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, 
                                         l_ids=l_ids, l_graphJSON=l_graphJSON,
                                         s_ids=s_ids, s_graphJSON=s_graphJSON,
                                         p_ids=p_ids, p_graphJSON=p_graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    one_label_idx = list(df.columns).index(ONE_LABEL_COL)
    classification_labels.insert(one_label_idx, 0)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
