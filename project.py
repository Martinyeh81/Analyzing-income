import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#SVM
data = pd.read_csv('adult_train.csv')


#preprocessing

one_hot_encoding = pd.get_dummies(data[['f2','f4','f6','f7','f8','f9','f10']])
data = data.drop(['f2','f4','f6','f7','f8','f9','f10'],1)
data = pd.concat([one_hot_encoding, data], axis=1)

#label change
def label_change(x):
    if x == ' <=50K':
        return '0'
    if x == ' >50K':
        return '1'
    if x == ' <=50K.':
        return '0'
    if x == ' >50K.':
        return '1'
#country change
def country_change(x):
    if x == ' United-States':
        return '0'
    else:
        return '1'

data['label'] = data['label'].apply(label_change)
data['f14'] = data['f14'].apply(country_change)

#inbalance
#thanks to Kaggle kernels for the following code[https://www.kaggle.com/lsjsj92/porto-simple-eda-with-python-unbalanced-data]
num_label_0, num_label_1 = data.label.value_counts()
label_0 = data[data['label'] == '0']
label_1 = data[data['label'] == '1']
label_1_over = label_1.sample(num_label_0, replace=True)
final_data_over = pd.concat([label_1_over, label_0], axis=0)

#split train and label
data_split = np.split(final_data_over, [-1], axis=1)

#Standardization
X_train = data_split[0]
X_train = X_train.astype(np.float)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)

#PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_train_std)
pca_data_final = pd.DataFrame(data= pca_data, columns=['f1','f2'])



#final_data= pd.concat([pca_data_final,data_split[1]['label']], axis=1)
#split train set test set

X_train, X_test, y_train, y_test = train_test_split(pca_data_final.values, data_split[1]['label'].values, test_size=.4,
                                                    random_state=42)

#SVM

model = SVC(kernel ='rbf',C=10, gamma=0.001)
model.fit(X_train, y_train)
predict_testing_label = model.predict(X_test)
testing_accuracy1 = accuracy_score(y_test, predict_testing_label)
print(testing_accuracy1)
testing_accuracy2 = model.score(X_test,y_test)
print(testing_accuracy2)

#cross_val_score

scores1 = cross_val_score(model, X_train, y_train, cv=10)
print(scores1.mean())
#find parameters
#thanks to Kaggle kernels for the following code[https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv]
SVM_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1,10,100]},{'kernel': ['linear'], 'C': [1,10,100]}]
scores = ['accuracy']
for score in scores:
    clf = GridSearchCV(SVC(), SVM_parameters, cv=5, scoring='%s' % score)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

#confusion matrix
predict_tr_label_cm = pd.Series(predict_testing_label, name='predict_testing_label')
con_matrix = pd.crosstab(y_test,predict_tr_label_cm,margins=True)
print(con_matrix)
print(classification_report(y_test,predict_testing_label))





