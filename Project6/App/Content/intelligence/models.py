
import os
import numpy as np
import pickle
from Content.Pkg.helper import basicHTMLTextCleaner # calls from run.py

#%%

def prediction_1(title,body):
    '''
    main call to API pipe

    Inputs
    ------
    title, body : str
        the title and the body of the question.

    Returns
    -------
    tag_hat : list
        list of predicted tags.
    qualityFlag : int
        0 if everything is ok,
        1 if body does not provide keywords,
        2 if title does not provide keywords,
        3 if neither body nor title provide keywords (i.e. will return the average answer from the model).
    '''

    pathToIntellDir = os.path.split(os.path.realpath(__file__))[0] # works if material is in same directory than this file
    
    #-- load needed material
    tokenizer = pickle.load(open(os.path.join(pathToIntellDir,'tokenizer.pkl'),'rb'))
    stopwords = pickle.load(open(os.path.join(pathToIntellDir,'stopwords.pkl'),'rb'))
    stemer = pickle.load(open(os.path.join(pathToIntellDir,'stemer.pkl'),'rb'))
    
    count_title = pickle.load(open(os.path.join(pathToIntellDir,'count_title.pkl'),'rb'))
    count_body = pickle.load(open(os.path.join(pathToIntellDir,'count_body.pkl'),'rb'))
    tfidf_title = pickle.load(open(os.path.join(pathToIntellDir,'tfidf_title.pkl'),'rb'))
    tfidf_body = pickle.load(open(os.path.join(pathToIntellDir,'tfidf_body.pkl'),'rb'))
    
    model = pickle.load(open(os.path.join(pathToIntellDir,'model.pkl'),'rb'))
    count_tag = pickle.load(open(os.path.join(pathToIntellDir,'count_tag.pkl'),'rb'))
    
    #-- data processing
    qualityFlag = 0

    # clean text
    title_c = basicHTMLTextCleaner(title,tokenizer,stopwords,stemer)
    body_c = basicHTMLTextCleaner(body,tokenizer,stopwords,stemer)
    # count frequencies
    title_f = count_title.transform([title_c])
    body_f = count_body.transform([body_c])
    # get TF-IDF
    title_i = tfidf_title.transform(title_f)
    body_i = tfidf_body.transform(body_f)
    # get X
    x = np.hstack((body_i.toarray(),title_i.toarray())) # SEE HOW TO AVOID .toarray()...

    if 0 == body_i.sum():
        if 0 == title_i.sum():
            qualityFlag = 3
        else:
            qualityFlag = 1
    elif 0 == title_i.sum():
        qualityFlag = 2

    #-- prediction
    y_hat = model.predict(x)
    tag_hat = count_tag.inverse_transform(y_hat)
    tag_hat = list(tag_hat[0])
    
    return tag_hat,qualityFlag
