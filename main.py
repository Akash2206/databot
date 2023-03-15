import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, recall_score
from sklearn.tree import DecisionTreeClassifier # decision tee algorithm for classification
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score    
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_models():
  models = dict()
  models['dt_ent'] = DecisionTreeClassifier(criterion='entropy')
  models['dt_gini'] = DecisionTreeClassifier(criterion='gini')
  models['mlr']=LogisticRegression()
  models['lsvc']=SVC(kernel='linear')
  models['rsvc']=SVC()
  models['ssvc']=SVC(kernel='sigmoid')
  models['psvc']=SVC(kernel='poly')
  
  return models

def cross_evaluator(model, X, y):
  scores = cross_val_score(model, X, y, cv=5)
  return np.mean(scores)

def run(df, target, test_size):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    print('------the following result is the accuracy of different model in the dictionary of SML based on 5-fold cross validation-----')
    for name, model in models.items():
        scores = cross_evaluator(model, X, y)
        results.append(scores)
        names.append(name)
    data_dict = {'Name': names, 'Score': results}
    output_df = pd.DataFrame(data_dict)
    return output_df