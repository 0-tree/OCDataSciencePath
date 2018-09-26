# -*- coding: utf-8 -*-
"""

Helper functions or Project 6


"""


#%%


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



#%% END




