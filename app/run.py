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

def tokenize_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# write custom scoring for multiclass classifier
def multi_class_score(y_true, y_pred):
    accuracy_results = []
    for i, column in enumerate(y_true.columns):
        accuracy = accuracy_score(
            y_true.loc[:, column].values, y_pred[:, i])
        accuracy_results.append(accuracy)
    avg_accuracy = np.mean(accuracy_results)
    return avg_accuracy


# load data
engine = create_engine('sqlite:///../data/etl_disaster.db')
df = pd.read_sql_table('message_table', engine)

# load model
model = joblib.load("../models/cv_t.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_top5 = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    top_5_response_counts = df_top5.sum(axis=0).sort_values(ascending=False)[:5]
    top_5_response_names = list(top_5_response_counts.index)

    
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
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        #The second visualizations - Top 5 response
        {
            'data': [
                Bar(
                    x=top_5_response_names,
                    y=list(top_5_response_counts)
                )
            ],

            'layout': {
                'title': 'Top 5 Responses',
                'yaxis': {
                    'title': "Number of Responses"
                },
                'xaxis': {
                    'title': "Kind of Response"
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()