import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
import pickle
import seaborn as sns
import joblib

df = pd.read_csv('/Users/rohankumar/Desktop/Data Science/Datasets/bike-sharing-demand/train.csv', index_col=0, header = 0)


x = df[['temp','humidity','windspeed']]
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

classifier = xgb.sklearn.XGBClassifier(nthread = -1, seed = 1)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')
rp = sns.regplot(x=y_test, y=predictions)


joblib.dump(classifier,'notebooks/xgb.json')



