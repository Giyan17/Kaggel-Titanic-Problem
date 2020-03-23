#Data preprocessing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('train.csv')

# Replace using mean #
mean = dataset1['Age'].mean()
dataset1['Age'].fillna(mean, inplace=True)


dataset1['Embarked'].fillna('S', inplace=True)


X = dataset1.iloc[:, [2,4,5,6,7,9,11]].values
y = dataset1.iloc[:, 1].values

dataset1.isnull().sum()


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
#onehotencoder = OneHotEncoder(categorical_features = [6])
#X = onehotencoder.fit_transform(X).toarray()

##################################################
    
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, y)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X, y)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)



#iMPORTING TEST SET
Dataset_test = pd.read_csv('test.csv')
Dataset_test.isnull().sum()

# Replace using mean #
mean = Dataset_test['Age'].mean()
Dataset_test['Age'].fillna(mean, inplace=True)
mean = Dataset_test['Fare'].mean()
Dataset_test['Fare'].fillna(mean, inplace=True)


dataset1['Embarked'].fillna('S', inplace=True)


X_test = Dataset_test.iloc[:, [1,3,4,5,6,8,10]].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6])
#onehotencoder = OneHotEncoder(categorical_features = [6])
#X_test = onehotencoder.fit_transform(X_test).toarray()

y_pred = classifier.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": Dataset_test["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('titanic_RandomForestedit.csv', index=False)

