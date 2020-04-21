import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Extract category_name proportions as a whole
    category_proportions = df.iloc[:, 4:].sum()/df.shape[0]
    category_names = list(df.columns[4:])
    
    # Extract category_name proportions based on three different genres
    category_sum_by_genre = df.groupby('genre').sum()
    ## genre = direct:
    category_direct_proportion = category_sum_by_genre.iloc[0, 1:]/genre_counts[0]
    ## genre = news:
    category_news_proportion = category_sum_by_genre.iloc[1, 1:]/genre_counts[1]
    ## genre = social:
    category_social_proportion = category_sum_by_genre.iloc[2, 1:]/genre_counts[2]
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_proportions
                )
            ],

            'layout': {
                'title': 'Proportion of Categories',
                'yaxis': {
                    'title': "Message proportions"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_direct_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Categories in Direct Genre',
                'yaxis': {
                    'title': "Message Proportions"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_news_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Categories in News Genre',
                'yaxis': {
                    'title': "Message Proportions"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_social_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Categories in Social Genre',
                'yaxis': {
                    'title': "Message Proportions"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
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