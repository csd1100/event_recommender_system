import numpy as np
import xlsxwriter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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
