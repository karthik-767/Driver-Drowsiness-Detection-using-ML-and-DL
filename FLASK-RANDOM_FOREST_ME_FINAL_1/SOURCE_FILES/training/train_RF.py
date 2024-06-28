import math
import numpy as np 
import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

seed = 42


# Load the dataset
data_csv1 = 'D:\\DDD SYS\\RANDOM_FOREST_ME\\SOURCE_FILES\\training\\ASP_RATIOS_TMP.csv'
df1 = pd.read_csv(data_csv1)
df1['EAR'] = df1['EAR'].astype(float)
df1['MAR'] = df1['MAR'].astype(float)
df1['PUC'] = df1['PUC'].astype(float)
df1['MOE'] = df1['MOE'].astype(float)
data_csv2 = 'D:\\DDD SYS\\RFOREST_NEW_DATA\\SOURCE_FILES\\training\\data_new.csv'
df2 = pd.read_csv(data_csv2)
df2 = df2.drop(['Frame_Num'], axis=1)
df2 = df2.drop(df2[df2['Label'] == 'distracted'].index)
df2 = df2.drop(['P_ID'], axis=1)
df = pd.concat([df1, df2], axis=0)
label_mapping = {'alert': 0, 'drowsy': 1}
df['Label'] = df['Label'].replace(label_mapping)

X = df.drop(['Label'], axis=1)
y = df['Label']
seed=42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify = y)


print(X_train.shape, X_test.shape)

clf = RandomForestClassifier(n_estimators=10, random_state=seed)
clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)

# confusion matrix

matrix = confusion_matrix(y_test,test_pred, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(y_test,test_pred,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(y_test,test_pred,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics

sensitivity = round(tp/(tp+fn), 3)
specificity = round(tn/(tn+fp), 3)
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3)
balanced_accuracy = round((sensitivity+specificity)/2, 3)

precision = round(tp/(tp+fp), 3)
f1Score = round((2*tp/(2*tp + fp + fn)), 3)

# Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
# A model with a score of +1 is a perfect model and -1 is a poor model

from math import sqrt

mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)


model_path=r"D:\DDD SYS\RANDOM_FOREST_ME\SOURCE_FILES\training\models\created_dataset_1.pkl"
with open(model_path, 'wb') as file:
    pickle.dump(clf, file)
