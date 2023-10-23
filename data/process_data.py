"""
This script loads data from CSV files, merges them, cleans the data, and saves the 
cleaned dataframe to SQLlite database.


Functions:
    load_data: Load data from CSV files.
    clean_data: Clean the data.
    save_data: Save the cleaned dataframe to a SQLlite database.
    main: Main function of the script.

Author:
   Conor Duffy
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files.

    Args:
        messages_filepath (str): The file path of the messages CSV file.
        categories_filepath (str): The file path of the categories CSV file.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    # print(messages.head())
    categories = pd.read_csv(categories_filepath)
    # print(categories.head())

    # merge datasets
    df = pd.merge(messages, categories, on="id")
    # print(df.head())
    return df


def clean_data(df):
    """
    Clean the data.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The cleaned dataframe.
    """

    clean_df = df.copy()

    # create a dataframe of the 36 individual category columns
    categories = clean_df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe to extract column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    # print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string and convert to numeric
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # drop the original categories column from `df`
    clean_df = pd.concat([clean_df.drop("categories", axis=1), categories], axis=1)

    # drop duplicates
    clean_df.drop_duplicates(inplace=True)

    return clean_df


def save_data(df, database_filename):
    """
    Save the cleaned dataframe to a SQLite database.

    Args:
        df (pandas.DataFrame): The cleaned dataframe.
        database_filename (str): The filename of the SQLite database.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("dr_messages_categorized", engine, index=False)


def main():
    """
    Main function to execute the data processing pipeline.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            f"Loading data...\n    MESSAGES: {messages_filepath}\n CATEGORIES:{categories_filepath}"
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
