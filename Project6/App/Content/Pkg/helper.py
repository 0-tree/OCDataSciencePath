# -*- coding: utf-8 -*-
"""

Helper functions or Project 6


"""


#%%


import numpy as np
from bs4 import BeautifulSoup # conda install beautifulsoup4
import nltk
#nltk.download('punkt')


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


#%%
    
def avg_jacard(y_true,y_pred):
    '''
    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics
    '''
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    
    return jacard.mean()


#%% END




