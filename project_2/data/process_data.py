# import libraries
import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads two csv and merge them into one.

    params:
    messages_filepath -- message filepath.
    categories_filepath -- categories filepath.

    output:
    merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    This function loads a dataframe a perform some cleaning steps.

    params:
    df -- input dataframe.

    output:
    cleaned dataframe
    """

    # obtain an object that contains the name of the categories in its corpus
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[0]
    row = categories.iloc[0]

    # extract the name of the categories from the corpus and use it to create column names
    category_colnames = row.apply(lambda x: x.split("-")[0]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])
        categories[column] = pd.to_numeric(categories[column], errors='coerce')

    categories['related'] = categories['related'].replace(to_replace=2, value=1)     # replace 2s by 1s
    # concat the category dataframe to the main dataframe. Drop duplicates.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    This function saves a dataframe into a database.

    params:
    df -- input dataframe.
    database_filename --  name of the database

    output:
    database

    """
    if os.path.exists(database_filename):
        os.remove(database_filename)
    engine = create_engine('sqlite:///' + database_filename)
    return df.to_sql('ft_messages', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
