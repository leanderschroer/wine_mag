import pandas as pd

def cnvt_clmn_words(df,name):
    if name not in df.columns.values:
        print ('Key Error:\''+str(name)+'\' is not a column')
        return None
    words = df[name].astype('str').str.lower()
    words = words.values
    words = ' '.join(words)
    words = words.split(' ')
    return pd.Series(words)

def get_words(df):
    raw_words=cnvt_clmn_words(df,'description')
    varieties=cnvt_clmn_words(df,'variety')
    #countries=cnvt_clmn_words(df,'country')
    words = raw_words[~raw_words.isin(varieties)]
    words = words.value_counts().sort_index()
    return words[words>50].index.values


    


