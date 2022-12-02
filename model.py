import pandas as pd
import re

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle  # pour stocker les fichiers exécutables
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("ockyoTrain.csv", delimiter=',')
print("Shape of the DataFrame:", df.shape)
df.reset_index(inplace=True, drop=True)
print(df.head())

# object of WordNetLemmatizer
lm = WordNetLemmatizer()


def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('french'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus


corpus = text_transformation(df['text'])

cv = CountVectorizer(ngram_range=(1, 2))
traindata = cv.fit_transform(corpus)
X = traindata

y = df.label
parametersDt = {

     'n_estimators': [500, 1000, 1500],
     'max_depth': [5, 10, None],
     'min_samples_split': [5, 10, 15],
     'min_samples_leaf': [1, 2, 5, 10],
     'bootstrap': [True, False]}

grid_search = GridSearchCV(RandomForestClassifier(),parametersDt,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(X, y)
grid_search.best_params_
for i in range(25):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])

DT =RandomForestClassifier(
                                      max_depth=grid_search.best_params_['max_depth'],
                                      n_estimators=grid_search.best_params_['n_estimators'],
                                      min_samples_split=grid_search.best_params_['min_samples_split'],
                                      min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                      bootstrap=grid_search.best_params_['bootstrap'])

DT.fit(X, y)

test_df = pd.read_csv('ockyoTest.csv', delimiter=',')
X_test, y_test = test_df.text, test_df.label

# pre-processing of text
test_corpus = text_transformation(X_test)
# convert text data into vectors
testdata = cv.transform(test_corpus)
# predict the target
predictionsDT = DT.predict(testdata)
print('Decision Tree Classifier')
acc_score = accuracy_score(y_test, predictionsDT)
pre_score = precision_score(y_test, predictionsDT)
rec_score = recall_score(y_test, predictionsDT)
print('Accuracy_score: ', acc_score)
print('Precision_score: ', pre_score)
print('Recall_score: ', rec_score)






def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = DT.predict(transformed_input)
    return prediction


pickle.dump(DT,open("model.pkl", 'wb'))