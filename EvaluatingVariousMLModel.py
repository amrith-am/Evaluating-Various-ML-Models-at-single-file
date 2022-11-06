#!/usr/bin/env python
# coding: utf-8

# # Evaluating Various ML Models at single file

# # Loading Dataset

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from pandas import read_csv


# In[ ]:


url = "kidny_stone.csv"
names = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']
dataset = read_csv(url, names=names)


# # Summarize Dataset

# In[ ]:


print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())


# # Visualize Data

# In[ ]:


# In[ ]:


dataset.plot(kind='bar', subplots=True, layout=(2, 2))
pyplot.title('BAR PLOT')
pyplot.show()

dataset.hist()
pyplot.title('HISTOGRAM PLOT')
pyplot.show()

scatter_matrix(dataset)
pyplot.title('SCATTER PLOT')
pyplot.show()


# # Evaluating various ML Algorithm

# In[ ]:


# 6 ML Algorithm


# In[ ]:


array = dataset.values
X = array[:, 0:6]
y = array[:, 6]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1, shuffle=True)


# In[ ]:


models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[ ]:


results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.ylim(.990, .999)
pyplot.bar(names, res, color='maroon', width=0.6)

pyplot.title('Algorithm Comparison')
pyplot.show()
