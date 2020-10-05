
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:33:25 2020

@author: Srishti
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

file= './data/GigBeee (Responses).xlsx'

xls = pd.ExcelFile(file)
gig = xls.parse('Form responses 1')

gig.columns = ['Name', 'Age', 'Gender', 'Qualification', 
                'SkillDevCourse', 'RealLifeAcadProjects', 'WorkEx', 'EdImpartsPracticalKnowledge',
                'SkillsforProfTask','SkillsOnPlatform','GigWorkers','SideWork','ProjectDuration',
                'DailyTime','FreelancerPast','FreelancerFuture','Charge','RegisterOnWebsite',
                'Reason']

X=gig[['Age', 'Gender', 'Qualification', 'SkillDevCourse', 'RealLifeAcadProjects']]# 'WorkEx']]
Y=gig[['EdImpartsPracticalKnowledge']]
X.head()
Y.head()
#EDA for checking how mant people said education imparted practical knowledge
print(gig.groupby(['EdImpartsPracticalKnowledge'])['Name'].count())
# We see 33 people say No , 60 people are not sure and 86 people say education imparted practical knowledge


from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
labelEncoder.fit(X['Gender'])
X['Gender'] = labelEncoder.transform(X['Gender'])
labelEncoder.fit(X['Qualification'])
X['Qualification'] = labelEncoder.transform(X['Qualification'])
labelEncoder.fit(X['SkillDevCourse'])
X['SkillDevCourse'] = labelEncoder.transform(X['SkillDevCourse'])
labelEncoder.fit(X['RealLifeAcadProjects'])
X['RealLifeAcadProjects'] = labelEncoder.transform(X['RealLifeAcadProjects'])
#labelEncoder.fit(X['WorkEx'])
#X['WorkEx'] = labelEncoder.transform(X['WorkEx'])

labelEncoder.fit(Y['EdImpartsPracticalKnowledge'])
Y['EdImpartsPracticalKnowledge']=labelEncoder.transform(Y['EdImpartsPracticalKnowledge'])



X=np.array(X)
Y=np.array(Y)
X
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))



 
import seaborn as seabornInstance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


X = gig.drop(['Name','RegisterOnWebsite'], axis=1)
y = gig[['RegisterOnWebsite']]


training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 50)

labelEncoder = LabelEncoder()
labelEncoder.fit(X['Gender'])
X['Gender'] = labelEncoder.transform(X['Gender'])
labelEncoder.fit(X['Qualification'])
X['Qualification'] = labelEncoder.transform(X['Qualification'])
labelEncoder.fit(X['SkillDevCourse'])
X['SkillDevCourse'] = labelEncoder.transform(X['SkillDevCourse'])
labelEncoder.fit(X['RealLifeAcadProjects'])
X['RealLifeAcadProjects'] = labelEncoder.transform(X['RealLifeAcadProjects'])
labelEncoder.fit(X['WorkEx'])
X['WorkEx'] = labelEncoder.transform(X['WorkEx'])
labelEncoder.fit(X['EdImpartsPracticalKnowledge'])
X['EdImpartsPracticalKnowledge'] = labelEncoder.transform(X['EdImpartsPracticalKnowledge'])
labelEncoder.fit(X['SkillsforProfTask'])
X['SkillsforProfTask'] = labelEncoder.transform(X['SkillsforProfTask'])
labelEncoder.fit(X['SkillsOnPlatform'])
X['SkillsOnPlatform'] = labelEncoder.transform(X['SkillsOnPlatform'])

labelEncoder.fit(X['GigWorkers'])
X['GigWorkers'] = labelEncoder.transform(X['GigWorkers'])
labelEncoder.fit(X['SideWork'])
X['SideWork'] = labelEncoder.transform(X['SideWork'])
labelEncoder.fit(X['ProjectDuration'])
X['ProjectDuration'] = labelEncoder.transform(X['ProjectDuration'])


#labelEncoder.fit(X['DailyTime'])
#X['DailyTime'] = labelEncoder.transform(X['DailyTime'])
labelEncoder.fit(X['FreelancerPast'])
X['FreelancerPast'] = labelEncoder.transform(X['FreelancerPast'])
labelEncoder.fit(X['FreelancerFuture'])
X['FreelancerFuture'] = labelEncoder.transform(X['FreelancerFuture'])
labelEncoder.fit(X['Charge'])
X['Charge'] = labelEncoder.transform(X['Charge'])
labelEncoder.fit(X['Reason'])
X['Reason'] = labelEncoder.transform(X['Reason'])

labelEncoder.fit(y['RegisterOnWebsite'])
y['RegisterOnWebsite'] = labelEncoder.transform(y['RegisterOnWebsite'])

X_Encoded=X
Y_Encoded=y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.25, random_state=0)
               
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
print(clf.score(X_test, y_test))    
    

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

import mglearn
from sklearn.neighbors import KNeighborsRegressor
# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,stratify=y,test_size=0.30)
# instantiate the model and set the number of neighbors to consider to 2
reg = KNeighborsRegressor(n_neighbors=2)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=200,test_size=0.2)
lr = LinearRegression().fit(X_train, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))



from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=200,test_size=0.2)
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

ridge35 = Ridge(alpha=30).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge35.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge35.score(X_test, y_test)))

plt.plot(ridge35.coef_, '^')
plt.plot(lr.coef_, 'o')
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.xlim(-0.002, 0.002)
plt.ylim(0.05, 0.12)
plt.legend()

from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-0.35, 0.15)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")



corr = X_Encoded.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

X=gig[['Age','Qualification']]
k=gig[['RegisterOnWebsite']]

labelEncoder.fit(X['Qualification'])
X['Qualification'] = labelEncoder.transform(X['Qualification'])

labelEncoder.fit(k['RegisterOnWebsite'])
k['RegisterOnWebsite'] = labelEncoder.transform(k['RegisterOnWebsite'])
X.feature_names=X.columns.values
X=np.array(X)
y=[]
for i in range(len(k)):
    y.append(k['RegisterOnWebsite'].values[i])
y=np.array(y)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=1.0, ax=ax, alpha=0.9)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

X_train, X_test, y_train, y_test = train_test_split(X_Encoded, y, stratify=y, random_state=0,test_size=0.1)
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


#now c=100
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))


#now c=0.001
logreg001 = LogisticRegression(C=0.001).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(X_Encoded.shape[1]), X_Encoded.columns.values, rotation=90)
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()

#Age and Qualfication show better correlation
#reallifeacademic project and skill

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred, normalize=False))


from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X_Encoded, y, test_size=0.2, random_state=12)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d",(X_test.shape[0], (y_test != y_pred).sum()))
  

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred, normalize=False))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
  
# Normalizing the data so that the data approximately  
# follows a Gaussian distribution 
X_normalized = normalize(X_scaled) 
  
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 


pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 

plt.figure(figsize =(8, 8)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward'))) 


ac2 = AgglomerativeClustering(n_clusters = 2) 
  
# Visualizing the clustering 
plt.figure(figsize =(6, 6)) 
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = ac2.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 

ac3 = AgglomerativeClustering(n_clusters = 3) 
  
plt.figure(figsize =(6, 6)) 
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac3.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 

ac4 = AgglomerativeClustering(n_clusters = 4) 

plt.figure(figsize =(6, 6)) 
plt.scatter(X_principal['P1'], X_principal['P2'], 
			c = ac4.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 


ac5 = AgglomerativeClustering(n_clusters = 5) 

plt.figure(figsize =(6, 6)) 
plt.scatter(X_principal['P1'], X_principal['P2'], 
			c = ac5.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 

ac6 = AgglomerativeClustering(n_clusters = 6) 

plt.figure(figsize =(6, 6)) 
plt.scatter(X_principal['P1'], X_principal['P2'], 
			c = ac6.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 


k = [2, 3, 4, 5, 6] 

# Appending the silhouette scores of the different models to the list 
silhouette_scores = [] 
silhouette_scores.append( 
		silhouette_score(X_principal, ac2.fit_predict(X_principal))) 
silhouette_scores.append( 
		silhouette_score(X_principal, ac3.fit_predict(X_principal))) 
silhouette_scores.append( 
		silhouette_score(X_principal, ac4.fit_predict(X_principal))) 
silhouette_scores.append( 
		silhouette_score(X_principal, ac5.fit_predict(X_principal))) 
silhouette_scores.append( 
		silhouette_score(X_principal, ac6.fit_predict(X_principal))) 

# Plotting a bar graph to compare the results 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 20) 
plt.ylabel('S(i)', fontsize = 20) 
plt.show()


X = gig.drop(['Name','RegisterOnWebsite'], axis=1)
y = gig[['RegisterOnWebsite']]

labelEncoder = LabelEncoder()
labelEncoder.fit(X['Gender'])
X['Gender'] = labelEncoder.transform(X['Gender'])
labelEncoder.fit(X['Qualification'])
X['Qualification'] = labelEncoder.transform(X['Qualification'])
labelEncoder.fit(X['SkillDevCourse'])
X['SkillDevCourse'] = labelEncoder.transform(X['SkillDevCourse'])
labelEncoder.fit(X['RealLifeAcadProjects'])
X['RealLifeAcadProjects'] = labelEncoder.transform(X['RealLifeAcadProjects'])
labelEncoder.fit(X['WorkEx'])
X['WorkEx'] = labelEncoder.transform(X['WorkEx'])
labelEncoder.fit(X['EdImpartsPracticalKnowledge'])
X['EdImpartsPracticalKnowledge'] = labelEncoder.transform(X['EdImpartsPracticalKnowledge'])
labelEncoder.fit(X['SkillsforProfTask'])
X['SkillsforProfTask'] = labelEncoder.transform(X['SkillsforProfTask'])
labelEncoder.fit(X['SkillsOnPlatform'])
X['SkillsOnPlatform'] = labelEncoder.transform(X['SkillsOnPlatform'])

labelEncoder.fit(X['GigWorkers'])
X['GigWorkers'] = labelEncoder.transform(X['GigWorkers'])
labelEncoder.fit(X['SideWork'])
X['SideWork'] = labelEncoder.transform(X['SideWork'])
labelEncoder.fit(X['ProjectDuration'])
X['ProjectDuration'] = labelEncoder.transform(X['ProjectDuration'])


#labelEncoder.fit(X['DailyTime'])
#X['DailyTime'] = labelEncoder.transform(X['DailyTime'])
labelEncoder.fit(X['FreelancerPast'])
X['FreelancerPast'] = labelEncoder.transform(X['FreelancerPast'])
labelEncoder.fit(X['FreelancerFuture'])
X['FreelancerFuture'] = labelEncoder.transform(X['FreelancerFuture'])
labelEncoder.fit(X['Charge'])
X['Charge'] = labelEncoder.transform(X['Charge'])
labelEncoder.fit(X['Reason'])
X['Reason'] = labelEncoder.transform(X['Reason'])

labelEncoder.fit(y['RegisterOnWebsite'])
y['RegisterOnWebsite'] = labelEncoder.transform(y['RegisterOnWebsite'])

X=np.array(X)
y=np.array(y)
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(criterion="entropy",max_depth=5, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["0", "1","2"], feature_names=['Age', 'Gender', 'Qualification', 'SkillDevCourse',
       'RealLifeAcadProjects', 'WorkEx', 'EdImpartsPracticalKnowledge',
       'SkillsforProfTask', 'SkillsOnPlatform', 'GigWorkers', 'SideWork',
       'ProjectDuration', 'DailyTime', 'FreelancerPast',
       'FreelancerFuture', 'Charge', 'Reason'], impurity=False, filled=True)

print("Feature importances:\n{}".format(tree.feature_importances_))

def plot_feature_importances_gig(model):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_gig(tree)



