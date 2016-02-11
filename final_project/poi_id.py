# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("../tools/")
from time import time

from tester import dump_classifier_and_data

from feature_format import featureFormat, targetFeatureSplit
from addTextAsFeature import updateMyDataset
from createNewFeatures import computeFraction
from StandardizeFeatures import StandardizeFeatures

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid

### open updated final_project_dataset, that already includes 
### new features 

with open("final_project_dataset_1feb.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
   
### commented section below shows steps taken to create new features
### uncommenting this part will output differend features from email data
    
### As it states in sklearn.decomposition.TruncatedSVD documentation
###
### SVD suffers from a problem called “sign indeterminancy”,
### which means the sign of the components_ and the output from
### transform depend on the algorithm and random state.
### To work around this, fit instances of this class to data once,
### then keep the instance around to do transformations.    
'''
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Load the dictionary containing email text data grouped by name
with open("to_poi_email_data.pkl","r") as text_file:
    text_dict = pickle.load(text_file)


#updateMyDataset add's features from email data (naming them x1,x2,x3,..xn)
# terms includes actual words corresponding to particular value
# of that feature.    
my_dataset, terms = updateMyDataset(text_dict,data_dict)

### Creating new features
for name in my_dataset:
    data_point = my_dataset[name]    
    
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages ) 
    data_point["fraction_from_poi"] = fraction_from_poi
    
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
'''   
#dump data for futher re-use
#SVD suffers from a problem called “sign indeterminancy”,
# which means the sign of the components_ and the output
# from transform depend on the algorithm and random state.
#pickle.dump( my_dataset, open("final_project_dataset_new.pkl", "w") )



  
my_dataset = data_dict


### Very high income from stocks and frequent interaction
### via emails with other poi's makes Derrick Jr. James
### features very similar to other persons of interest.
my_dataset.pop('DERRICK JR. JAMES V')
my_dataset.pop('TOTAL')
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK')
#feature names
features_list = ['poi','bonus',
 'exercised_stock_options',
 'expenses','fraction_to_poi',
 'restricted_stock','salary',
 'shared_receipt_with_poi',
 'total_payments',
 'x6','x8','x12','x34','x58']

                
### normalize slected features
my_dataset = StandardizeFeatures(my_dataset,features_list)


### convert dataset in to ndarray 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### split the data to perfrom cross validation
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.33, random_state=0)


Tree_parameters = {'criterion' : ('gini', 'entropy'),
              'max_features':('sqrt','log2')}
KNN_parameters = {'n_neighbors':[2,3,4,5,6,7],'weights':('uniform','distance'),
              'leaf_size':[1,2,3]}
SVC_parameters = [{'kernel': ['rbf'], 'gamma': [0.1,1e-2,1e-3, 1e-4],
                     'C': [1, 10, 100, 1000,1200]},
                   #  'class_weight':[{True: 1.4},{True: 1.2},{True: 1.3}] },
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000,1200],
                     },
                    {'kernel':['sigmoid'],'shrinking':[True,False],
                    'class_weight':[{True: 1.4},{True: 1.2},{True: 1.3}],
                    'C': [1, 10, 100, 1000,1200],
                     'gamma': [1e-2,1e-3, 1e-4],
                     'coef0':[0,0.3,0.6,1,1.5,10]},
                    {'kernel':['poly'],
                    'class_weight':[{True: 1.4},{True: 1.2},{True: 1.3}],
                    'C': [1, 10, 100, 1000,1200],'degree':[2,3],
                     'gamma': [1e-2,1e-3, 1e-4],
                     'coef0':[0,0.3,0.6,1,1.5,10]}]
                    
Ada_parameters = {'base_estimator':(GaussianNB(),DecisionTreeClassifier()),
                    'n_estimators':[10,20,30,40,50],
                    'learning_rate':[0.3,0.6,1,1.3,1.6,2]}  
                
Forest_parameters = {'n_estimators':[10,20,30,40,50],'criterion' : ('gini', 'entropy'),
                     'max_features':('sqrt','log2'),'max_depth':[2,3,4,5,6,7,8,9,10]}
                     
GradientBoost_param = {'n_estimators':[10,20,30,40,50,100],'loss':('deviance', 'exponential'),
                       'max_depth':[2,3,4,5,6,7,8,9,10],'max_features':('sqrt','log2'),
                       'learning_rate':[0.3,0.6,1,1.3,1.6,2]}
                       
Centroid_param = {'shrink_threshold':[-10,-7,-4,-3.6,-3.3,-3,-2.7,-2,-1,0,1,2,3]}

parameters = [
              #Ada_parameters,Forest_parameters,              
              KNN_parameters,
              Centroid_param,
              SVC_parameters]

names = [
         #"AdaBoostClassifier","RandomForestClassifier",
         "KNeighborsClassifier",
         "NearestCentroid","SVC"]
    

classifiers = [
              # AdaBoostClassifier(),
              # RandomForestClassifier(),
               KNeighborsClassifier(),
               NearestCentroid(),SVC()]               
            





for name, classifier, param in zip(names, classifiers,parameters):
    
    
    t0 = time()
    clf = GridSearchCV(classifier, param)
    clf.fit(features_train,labels_train)
    

    precision = precision_score(labels_test,clf.predict(features_test))
    recall = recall_score(labels_test,clf.predict(features_test))
    confusion_M = confusion_matrix(labels_test,clf.predict(features_test))

    print "##################################"
    print clf.best_params_

    print name,":","precision score", precision
    print name,":","recall score", recall
    print name,":", "confusion matrix", confusion_M
    print "done in %0.3fs" % (time() - t0)
    print "##################################"


### dump best perfoming classifier for testing
clf = SVC(kernel='sigmoid',C=10,coef0=0.5,gamma=0.01)
#clf = SVC(kernel='poly',C=1,coef0=1,gamma=0.01,class_weight= {True:1.4})
dump_classifier_and_data(clf, my_dataset, features_list)