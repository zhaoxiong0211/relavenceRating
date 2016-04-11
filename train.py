import numpy as np
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

train = pd.read_csv("./Desktop/dataC/hd/train.csv",encoding = 'iso-8859-1')
description = pd.read_csv("./Desktop/dataC/hd/product_descriptions.csv",encoding = 'iso-8859-1')
test = pd.read_csv("./Desktop/dataC/hd/test.csv",encoding = 'iso-8859-1')

# merge test and train to process data in the same time
allData = pd.concat((train, test), axis=0, ignore_index=True)
train_length = train.shape[0]

# tokenize
toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)

def tokenize(text):
    lowers = text.lower()
    tokens = toker.tokenize(lowers)
    res = []
    for word in tokens:
        res.extend(re.split('(\d+)',word))
    return [x for x in res if x]

allData['search_term'] = allData['search_term'].apply(lambda x:tokenize(x))
allData['product_title'] = allData['product_title'].apply(lambda x:tokenize(x))
description['product_description'] = description['product_description'].apply(lambda x:tokenize(x))

# find stem
def stemming(words):
    return [stemmer.stem(word) for word in words]

allData['search_term'] = allData['search_term'].apply(lambda x:stemming(x))
allData['product_title'] = allData['product_title'].apply(lambda x:stemming(x))
description['product_description'] = description['product_description'].apply(lambda x:stemming(x))

# combine description and dataset together
def description_convert(productId):
    return description.loc[description["product_uid"]==productId,"product_description"].values[0]

allData["product_description"] = allData["product_uid"].apply(lambda x:description_convert(x))

# count word match
def count_match(idx, case):
    words_1 = allData.loc[allData["id"]==idx,"search_term"].values[0]
    if case == 1:
        words_2 = allData.loc[allData["id"]==idx,"product_title"].values[0]
    else:
        words_2 = allData.loc[allData["id"]==idx,"product_description"].values[0]
    res = 0.0
    for word in words_1:
        if word in words_2:
            res += 1
    return res

allData["count_in_title"] = allData["id"].apply(lambda x: count_match(x,1))
allData["count_in_description"] = allData["id"].apply(lambda x: count_match(x,2))

allData["search_term_length"] = allData["search_term"].apply(lambda x: len(x)).astype(np.int64)

# train and test
all_train = allData.iloc[:train_length]
all_test = allData.iloc[train_length:]
predictors = ["count_in_title", "count_in_description", "search_term_length"]

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(all_train[predictors], all_train["relevance"])
y_pred = clf.predict(all_test[predictors])

# save result
pd.DataFrame({"id": all_test["id"], "relevance": y_pred}).to_csv('./Desktop/dataC/hd/submission.csv',index=False)
