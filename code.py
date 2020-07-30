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
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

cv = CountVectorizer()
event_list = []
for event in test['Event'].values:
    event_dict = {}
    i = 0
    feature_list = []
    cosine_sim_list = []
    print((event))
    input_feature = (model.predict(count_vect.transform([event.lower()]))[0])
    for combined_feature in employee_data['Combined']:
        feature_list.append(combined_feature)
        feature_list.append(input_feature)
        count_matrix = cv.fit_transform(feature_list)
        cosine_sim = cosine_similarity(count_matrix)
        cosine_sim_list.append([cosine_sim[0], i])
        i += 1
        feature_list = []

    event_dict['Event'] = event
    event_dict['Employees'] = ''
    sorted_list = sorted(cosine_sim_list, key=lambda x: x[0][1], reverse=True)
    min_score = 0.60
    for x in sorted_list:
        score = x[0][1]
        ind = x[1]
        if score < min_score:
            break
        else:
            print(employee_data['Name'][ind])
            if(len(event_dict['Employees']) < 1):
                event_dict['Employees'] = employee_data['Name'][ind]
            else:
                event_dict['Employees'] = event_dict['Employees'] + \
                    ',' + employee_data['Name'][ind]
    event_list.append(event_dict)
    print('--------------------------------------------------------------------')

workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()

Events = [x['Event'] for x in event_list]
Employees = [x['Employees'] for x in event_list]

worksheet.write(0, 0, 'Event')
worksheet.write(0, 1, 'Employees')
worksheet.write_column(1, 0, Events)
worksheet.write_column(1, 1, Employees)

workbook.close()
