import numpy as np
import xlsxwriter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2


def employee_data_combine_features(row):
    return row['Domain']+' '+row['Event1']+' '+row['Event2']


df = pd.read_csv('events.csv')
df['Type'] = df['Type'].str.lstrip()

employee_data = pd.read_csv('CCMLEmployeeData.csv')
employee_data['Combined'] = employee_data.apply(
    employee_data_combine_features, axis=1)

test = pd.read_csv('test.csv')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df['Event']).toarray()
labels = df['Type']

event_dict = dict(df.values)

N = 2
for event, types in event_dict.items():
    features_chi2 = chi2(features, labels == types)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

X_train, X_test, y_train, y_test = train_test_split(
    df['Event'], df['Type'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
