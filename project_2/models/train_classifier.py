import sys
import os
import nltk
from pandas.io import pickle
from sklearn.multiclass import OneVsRestClassifier

nltk.download(['punkt', 'wordnet', 'stopwords'])
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    This function loads data from a SQL dataframe and extracts features and target dataframes, and the names of the categories.

    params:
    database_filepath -- SQL database file

    output:
    X -- Features dataframe
    Y -- Target dataframe
    category_names -- categories from category column. This will be the target for the model.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    # create X, Y, and category names.
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    This function transforms a text to a tokenize array.

    params:
    database_filepath -- SQL database file

    output:
    X -- Features dataframe
    Y -- Target dataframe
    category_names -- categories from category column. This will be the target for the model.
    """
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    This functions crates the classification model. In this case a Random Forest Classifier is used
     to perform a multioutput classification.
     The output is the model itself.
    """

    # create the pipeline
    pipeline = Pipeline([('vectorized', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('classification', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))])

    # search hyperparameters
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'tfidf__use_idf': [True, False]}
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=4)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluate the model built in build_model function.

    params:
    model -- Model built in build_model function.
    X_test -- Features from testing dataframe.
    Y_test -- targets from testing dataframe.
    category_names -- categories from category column. This will be the target for the model.

    output:
    print report for each of the targets.
    """
    # predict
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    This function saves the model in pickle file

    params:
    model -- model after training and testing
    model_filepath -- path to locate the model.
    """
    if os.path.exists(model_filepath):
        os.remove(model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
