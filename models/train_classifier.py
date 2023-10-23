"""
This script loads data from a database, tokenizes text data, builds a classification model
, trains the model,evaluates the model's performance, and saves the trained model 
to a pickle file.

Functions:
    load_data: Load data from database.
    tokenize: Tokenize text data.
    save_model: Save the trained model to a pickle file.
    build_model: Build the classification model.
    main: Main function of the script.

Classes:
    None

Author:
   Conor Duffy 
"""
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


nltk.download(["punkt", "wordnet"])


def load_data(database_filepath):
    """
    Load data from database.

    Args:
        database_filepath (str): The filepath of the database.

    Returns:
        tuple: A tuple containing the features (X) and the targets (Y).
    """
    # load data from database
    engine = create_engine(f"sqlite:///{database_filepath}")
    # read sql table into dataframe df
    df = pd.read_sql_table("dr_messages_categorized", engine)
    x = df.loc[:, "message"].values
    y = df.iloc[:, 4:]
    category_names = y.columns
    return x, y, category_names


def tokenize(text):
    """
    Tokenize text data.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of clean tokens.

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a Random Forest Classifier model for multi-output classification.

    Returns:
        object: A Random Forest Classifier model to be trained.
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        "tfidf__use_idf": (True, False),
        # 'clf__estimator__n_estimators': [50, 100, 200],
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # Create GridSearch object and fit it to the data
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    Evaluate the performance of the trained model on the test data.

    Args:
        model (object): The trained model.
        X_test (array-like): The test data.
        Y_test (array-like): The true labels for the test data.
        category_names (list): The list of category names.
    """
    # Predict on the test data
    y_pred = model.predict(x_test)

    # Calculate the accuracy for each of them.
    for i, column in enumerate(y_test.columns):
        accuracy = accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])
        print(f"Category: {column}")
        print(f"Accuracy: {accuracy:.2f} \n")

    # Now we create a classification report for each of the 36 categories
    for i, category_name in enumerate(category_names):
        print(
            "Category:",
            category_name,
            "\n",
            classification_report(y_test.iloc[:, i], y_pred[:, i]),
        )

    # Overall accuracy
    accuracy = (y_pred == y_test).mean().mean()
    print(f"Overall Accuracy: {accuracy*100:.2f}% \n")


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Args:
        model (object): The trained model.
        model_filepath (str): The filepath to save the model.
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    """
    Main function to execute the training and evaluation of the model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(x_train, y_train)

        print("Evaluating model...")
        evaluate_model(model, x_test, y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
