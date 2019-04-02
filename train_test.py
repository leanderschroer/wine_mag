import multiprocessing as mp
import time
from typing import Counter
import get_names
import pandas as pd

def format_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        return get_names.tokenize_except(data['description'], data['variety'])
    except:
        print('Error: Bad column names')


def read_assemble(chunksize: int, nrows: int) -> pd.DataFrame:
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
        new_data = pd.concat(result, ignore_index=True, sort=False)

        return new_data
    return None


s = time.time()
assembled_data = read_assemble(700, 1400)
print(time.time() - s)
if assembled_data is not None:
    print(assembled_data.head())

# train = data.sample(frac=.8, axis = 0)
# test = data.loc[~data.index.isin(train.index)]
# model = DecisionTreeClassifier(max_depth = 5, criterion='entropy')
# model.fit(train.iloc[:,16:],train['variety'])
# predictions = model.predict(test.iloc[:,16:])
# dot_data = tree.export_graphviz(model, label='all',  out_file=None, filled=True, rounded=True, special_characters=True, class_names=varieties, feature_names=train.iloc[:,16:].columns.values) 
# graph = graphviz.Source(dot_data) 
# graph.render("wine")
