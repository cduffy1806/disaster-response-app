"""
This module is used to run the web application for the Disaster Response Project. 

The application uses a trained model to classify disaster messages and displays the results in a web interface. 

Functions:
    tokenize(text): Tokenizes the input text by converting it into lower case, stripping whitespace, and lemmatizing each word.
    index(): The main route of the web application. It displays the homepage with visuals and receives user input text for model.
    go(): The route that handles user query and model prediction. It displays the classification results of the user input text.
    main(): The main function that runs the web application server.
"""
import json
import plotly
import joblib
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """
    Tokenize the text by converting it into lower case, stripping whitespace, and lemmatizing each word.

    Parameters:
    text (str): The text to be tokenized.

    Returns:
    list: A list of the tokenized text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("dr_messages_categorized", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """
    This function is the main entry point for the web application. It renders the index page with visuals and
    receives user input text for the model to classify.
    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # Count the number of each category
    category_counts = df.iloc[:,4:].sum().sort_values(ascending=False).head(10)
    category_names = list(category_counts.index)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        }
        ,{
            "data": [Bar(x=category_names, y=category_counts.values)],
            "layout": {
                "title": "Top 10 Message Categories by Count",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        }
        ,{
            "data": [
                {
                    "x": category_names[:10],
                    "y": [df[df['genre'] == genre][category].sum() for category in category_names[:10]],
                    "name": genre,
                    "type": "bar"
                } for genre in genre_names
            ],
            "layout": {
                "title": "Stacked Genre Count for Top 10 Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
                "barmode": "stack"
            },
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    """
    This function handles the user query and displays the model results.
    It extracts the user query from the request arguments, uses the model to predict the classification for the query,
    and then renders the 'go.html' template with the query and classification result.
    """

    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    """
    This is the main function that runs the web application server.
    """
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
