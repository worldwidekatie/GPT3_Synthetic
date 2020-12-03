import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('imbd/data/imbd_orig.csv')
df = df.rename(columns={'review': 'text', 'sentiment':'label'})


"""Truncating"""
var1 = list(df["text"])
var = []
for i in var1:
    if len(i) > 900:
        var.append(i[:900])
    else:
        var.append(i)

length = []

for i in var:
    length.append(len(i))

df["length"] = length
print(df["length"].describe())

length = sorted(length, reverse=True)
print(length[:50])

df['text'] = var

df = df[['text', 'label']]
df.to_csv('imbd/data/trunc_imbd.csv', index=False)
print(df.describe())

"""Taking a smaller samples of the IMBD dataset"""

"""Test/Train split"""
df_train, df_test= train_test_split(df, test_size=0.33, random_state=55)

df_train.to_csv('imbd/data/imbd_train.csv', index=False)
df_test.to_csv('imbd/data/imbd_test.csv', index=False)

"""Getting 5% Seeds"""
pos = df_train[df_train['label'] == 1]
neg = df_train[df_train['label'] == 0]
print(neg.shape)
print(pos.shape)

pos_seed_05 = pos.sample(frac=.05, replace=False, random_state=79)
print(pos_seed_05.shape)
pos_seed_05.to_csv("imbd/data/pos_seed_05.csv", index=False)

neg_seed_05 = neg.sample(frac=.05, replace=False, random_state=79)
print(neg_seed_05.shape)
neg_seed_05.to_csv("imbd/data/neg_seed_05.csv", index=False)