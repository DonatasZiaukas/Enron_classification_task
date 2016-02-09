# -*- coding: utf-8 -*-


import sys

sys.path.append("../tools/")


import  numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))
stopset.update(['cc','enron','make','subject','new','way','hope','im','just','date',
                'month','monday','aa','pm','current','point','let','sent','goal','great',
                'like','today','team','want','use','time','week','ani','pleas','sure',
                'good','forward','would','jeff','well','know','thank','busi','origin'])


def convertToNdarray(textD):
    '''
    Dictionary keys and values are transfromed in numpy.ndarray for 
    indexing purposes
    '''
    matrix = []
    for k,v in textD.items():
        
        tmp_l = []
        tmp_l.append(k)
        tmp_l.append(v)
        matrix.append(np.array(tmp_l))
    
    return np.array(matrix)
    
def updateMyDataset(textD,dataset):
    '''
    TfidfVectorizer used to assign numerical values to the words and
    linear dimensionality reduction performed with sklearn 
    TruncatedSVD to include only first 100 components
    '''
    data = convertToNdarray(textD)
    text = data[:,1]
    
    #assigning numerical value to words in text data
    vectorizer = TfidfVectorizer(sublinear_tf=False,ngram_range=(1, 1),
                                 max_df=0.75,smooth_idf=False,
                                 stop_words=stopset,norm='l2')
     
                               
    features = vectorizer.fit_transform(text)
    
    #performing linear dimensionality reduction    
    lsa = TruncatedSVD(n_components=100,n_iter=10)
    lsa.fit(features)
    # terms_list is created for feature exploration by seeing 
    # actual words of the component
    terms = vectorizer.get_feature_names()
    terms_list = []
    for i, comp in enumerate(lsa.components_):
        termsInComp = zip(terms,comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:15]
        terms_list.append(sortedTerms)
        
    features_after_svd = lsa.transform(features)
    
    for name in dataset:
        
        index = np.where(data == name)[0]
        index = int(index)
        for i in range(len(features_after_svd[index][:])):
            #
            dataset[name]['x' + str(i)] = features_after_svd[index][i]
            
    return dataset, terms_list


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    