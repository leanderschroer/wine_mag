import collections
import multiprocessing as mp
import time

import pandas as pd

import get_names

# import graphviz


data = pd.read_csv('resources/winemag-data-130k-v2.csv', usecols=['variety', 'description'], nrows=100)


# column_list = get_names.get_words(data)

def transf_counts(df):
    counts = df['description'].str.split(' ')
    counts = counts.map(collections.Counter)
    counts = counts.to_dict()
    counts = pd.DataFrame(counts).transpose()
    new_data = pd.concat([df[['variety']], counts], sort=False)
    revealing_words = get_names.cnvt_clmn_words(df, 'variety')
    new_data = new_data[~new_data.index.isin(revealing_words)]
    return new_data


def read_assemble(chunksize, nrows):
    if __name__ == '__main__':
        reader = pd.read_csv('resources/winemag-data-130k-v2.csv', usecols=['variety', 'description'],
                             chunksize=chunksize, nrows=nrows)
        pool = mp.Pool()
        funclist = []
        for df in reader:
            f = pool.apply_async(transf_counts, [df])
            funclist.append(f)

        result = []
        for f in funclist:
            result.append(f.get(timeout=60))

        new_data = pd.concat(result, ignore_index=True, sort=False)
        return new_data
    return None


s = time.time()
read_assemble(700, 5000)
print(time.time() - s)

# train = data.sample(frac=.8, axis = 0)
# test = data.loc[~data.index.isin(train.index)]
# model = DecisionTreeClassifier(max_depth = 5, criterion='entropy')
# model.fit(train.iloc[:,16:],train['variety'])
# predictions = model.predict(test.iloc[:,16:])
# dot_data = tree.export_graphviz(model, label='all',  out_file=None, filled=True, rounded=True, special_characters=True, class_names=varieties, feature_names=train.iloc[:,16:].columns.values) 
# graph = graphviz.Source(dot_data) 
# graph.render("wine")
