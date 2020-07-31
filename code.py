import numpy as np
import xlsxwriter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def employee_data_combine_features(row):
    return row['Domain']+' '+row['Event1']+' '+row['Event2']


def lower_events(row):
    row = row['Event'].lower().lstrip()
    return row


def lstrip_types(row):
    row = row['Type'].lstrip()
    return row


df = pd.read_csv('events.csv')

employee_data = pd.read_csv('CCMLEmployeeData.csv')
employee_data['Combined'] = employee_data.apply(
    employee_data_combine_features, axis=1)

test = pd.read_csv('input.csv')

df['Event'] = df.apply(lower_events, axis=1).to_frame()[0]
df['Type'] = df.apply(lstrip_types, axis=1).to_frame()[0]
# test['Event'] = test.apply(lower_events, axis=1).to_frame()[0]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df['Event']).toarray()
labels = df['Type']

event_dict = dict(df.values)

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
    print(input_feature)
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
