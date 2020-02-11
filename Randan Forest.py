import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Random Forest
data = pd.read_csv('adult_train.csv')

#preprocessing

data['f2']=data['f2'].astype('category').cat.codes
data['f4']=data['f4'].astype('category').cat.codes
data['f6']=data['f6'].astype('category').cat.codes
data['f7']=data['f7'].astype('category').cat.codes
data['f8']=data['f8'].astype('category').cat.codes
data['f9']=data['f9'].astype('category').cat.codes
data['f10']=data['f10'].astype('category').cat.codes
data['f14']=data['f14'].astype('category').cat.codes
data['label']=data['label'].astype('category').cat.codes

#inbalance
#thanks to Kaggle kernels for the following code[https://www.kaggle.com/lsjsj92/porto-simple-eda-with-python-unbalanced-data]
num_label_0, num_label_1 = data.label.value_counts()
label_0 = data[data['label'] == 0]
label_1 = data[data['label'] == 1]
label_1_over = label_1.sample(num_label_0, replace=True)
final_data_over = pd.concat([label_1_over, label_0], axis=0)

#split train and label
data_split = np.split(final_data_over, [-1], axis=1)


#split train and test
x_train, x_test, y_train, y_test = train_test_split(data_split[0].values, data_split[1]['label'].values, test_size=0.4,
                                                    random_state=42)

#find parameter
#thanks to Kaggle kernels for the following code[https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv]
rfc=RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}
gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)



#Model

model = RandomForestClassifier(criterion='gini',n_estimators=500,
                                max_depth =8,
                               max_features='auto',
                                )
model.fit(x_train,y_train)
predit = model.predict(x_test)
accuray = accuracy_score(y_test,predit)
print(accuray)

predict_tr_label_cm = pd.Series(predit, name='predict_training_label')
con_matrix = pd.crosstab(y_test,predict_tr_label_cm,margins=True)
print(con_matrix)
print(classification_report(y_test,predit))
