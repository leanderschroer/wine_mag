import multiprocessing as mp
import time
from typing import Tuple, List, Any

import gc
import graphviz
import numpy
import pandas as pd
import scipy
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import get_names


def format_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['description'] = data['description'].apply(get_names.count_tokens)
        return data
    except:
        print('Error: Unspecified')


def read_assemble(chunksize: int, nrows: int) -> Tuple[Any, pd.Series, List]:
    if __name__ == '__main__':
        reader = pd.read_csv('resources/winemag-data-130k-v2.csv',
                             usecols=['variety', 'description'],
                             chunksize=chunksize, nrows=nrows)

        pool = mp.Pool()
        funclist = []
        for df in reader:
            f = pool.apply_async(format_data, [df])
            funclist.append(f)

        result = []
        for f in funclist:
            result.append(f.get(timeout=60))

        pool.close()
        pool.join()
        gc.collect()
        new_data = pd.concat(result, ignore_index=True, sort=False)

        #avoid returning dense matrix, it will flood your memory
        v = DictVectorizer()
        return (v.fit_transform(new_data['description']), new_data['variety'], v.get_feature_names())
    return None



def split_train_accuracy(X: object, y: pd.Series, test_size: float, random_state: int, depth: int) -> Tuple[object,float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    model = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    model.fit(X_train, y_train.astype(str))
    predictions = model.predict(X_test)
    accuracy = numpy.mean(y_test == predictions)
    return model, accuracy

if __name__ == '__main__':
    print('Assembling Dataset at '+str(time.time()))
    X, y, feature_names = read_assemble(1000, 13000)
    print('Training Model at '+str(time.time()))
    model, score = split_train_accuracy(X,y.astype(str),0.33,None,10)
    dot_data = tree.export_graphviz(model, label='all', out_file=None, filled=True, rounded=True,
                                    special_characters=True, class_names=list(set(feature_names)),
                                    feature_names=feature_names)
    graph = graphviz.Source(dot_data)
    graph.render("wine")