# -*- coding: utf-8 -*-
"""

Helper functions for Project 6


"""


#%%


import numpy as np
from bs4 import BeautifulSoup # conda install beautifulsoup4
import nltk
#nltk.download('punkt')
import matplotlib.pyplot as plt


#%% visualisation
# ---------------

def plotCV(cvmodel,alpha=3):
	'''
	basic plot of the mean and confidence intervals scores from the tests in CV.

	Inputs
	------
	cvmodel : sklearn.model_selection._search.GridSearchCV or equivalent
		fitted CV model, must contain a .cv_results_ attribute
	alpha : float or int, default 3
		number of stds to represent the confidence intervals.
	'''
	f = plt.figure(figsize=(20,5))
	f.add_subplot(111)
	mean = cvmodel.cv_results_['mean_test_score']
	std = cvmodel.cv_results_['std_test_score']
	plt.scatter(np.arange(len(mean)),mean,s=None,c='k',marker='o')
	plt.scatter(np.arange(len(mean)),mean+alpha*std,s=None,c='r',marker='^')
	plt.scatter(np.arange(len(mean)),mean-alpha*std,s=None,c='r',marker='v')
	plt.show()



#%% text cleaning
# ---------------

#%%


def basicHTMLTextCleaner(htmlText,tokenizer,stopwords,stemer,outputType='str'):
    '''
    basic tokenizer for raw html text.
    
    Inputs
    ------
    htmlText : str
        html text.
    tokenizer : nltk.tokenize.regexp.RegexpTokenizer or similar
        used to tokenize.
    stopwords : nltk.corpus.reader.wordlist.WordListCorpusReader or similar
        used to remove stopwords.
    stemer : nltk.stem.porter.PorterStemmer or similar
        used to stem.
    outputType : str ('str','Text')
        if 'str', returns a paragraph joined with ' ';
        if 'Text', returns an nltk.Text.    
        
    Returns
    -------
    text : str or nltk.Text
        the cleaned text.
    '''
    soup = BeautifulSoup(htmlText,features='html5lib')
    raw = soup.get_text(strip=False).lower()                # text from html + lower
    tokens = tokenizer.tokenize(raw)                        # tokenization
    tokens_stop = [t for t in tokens if t not in stopwords] # stopwords
    tokens_stp_stem = [stemer.stem(t) for t in tokens_stop] # stemming

    if outputType == 'str':
        text = ' '.join(tokens_stp_stem)
    elif outputType == 'Text':
        text = nltk.Text(tokens_stp_stem)
    else:
        raise ValueError('unknown outputType: {}'.format(outputType))
        
    return text


#%%
    
def basicTagTextCleaner(tagText,outputType='str'):
    '''
    basic tokenizer for tag text.
    
    Inputs
    ------
    tagText : str
        list of tags.
    outputType : str ('str','Text')
        if 'str', returns a paragraph joined with ' ';
        if 'Text', returns an nltk.Text.    
        
    Returns
    -------
    text : str or nltk.Text
        the cleaned tag text.
    '''
    tokens = tagText.split()
    
    if outputType == 'str':
        text = ' '.join(tokens)
    elif outputType == 'Text':
        text = nltk.Text(tokens)
    else:
        raise ValueError('unknown outputType: {}'.format(outputType))

    return text



#%% losses and scores
# -------------------

#%%
    
def avg_jacard(y_true,y_pred):
    '''
    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    '''
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    
    return jacard.mean()

#%%

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    
    Note
    ----
    It is the same as avg_jacard() :)
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)




#%% one-shot utilities
# --------------------

#%%
    
def isValidData(x,y,V_body):
    '''
    checks whether observations meet none of the following conditions
    - no word is present amongst the body features
    - no word is present amongst the title features
    - no tag is present amongst the tag features.
    
    Inputs
    ------
    x : np.array
        features data, with body features then title features.
    y : np.array
        multi-one-hot encoded target tags.
    V_body : int
        number of features for the body, used to derive the position of the features in x.
        
    Returns
    -------
    isValid : list of bool
        True if the observation is valid. Same length as x.shape[0]
        
    Note
    ----
    Current implementation heavily relies on x being of the form
    x = body features | title features.
    '''
    
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y have different nb of rows: {}'.format(x.shape[0],y.shape[0]))
    
    isValidBody = x[:,:V_body].max(axis=1) > 0
    isValidTitle = x[:,V_body:].max(axis=1) > 0
    isValidTag = y.max(axis=1) > 0
    
    isValid = list(isValidBody & isValidTitle & isValidTag)
    return isValid



#%% END




