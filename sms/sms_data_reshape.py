import pandas as pd 
from sklearn.model_selection import train_test_split

"""Cleaning up the original"""
# df = pd.read_csv('text_spam/spam_texts_orig.csv', encoding='latin')
# df = df[['v2', 'v1']]
# df = df.rename(columns={'v1': 'label', 'v2': 'texts'})
# df = df.replace({"spam": 0, "spam": 1})
# var1 = list(df["texts"])
# var = []
# for i in var1:
#     if len(i) > 200:
#         var.append(i[:200])
#     else:
#         var.append(i)

# length = []
# for i in var:
#     length.append(len(i))

# df["length"] = length
# print(df["length"].describe())

# length = sorted(length, reverse=True)
# print(length[:50])

# df['text'] = var

# df = df[['text', 'label']]
# df.to_csv('spam_texts.csv', index=False)




"""Taking a smaller sample of the full spam texts dataset"""
df = pd.read_csv('spam_texts_full.csv', encoding='latin')

# df = df.sample(n=2000, replace=False, random_state=79)

# df.to_csv('spam_texts.csv', index=False)

"""Test/Train split"""
df_train, df_test= train_test_split(df, test_size=0.33, random_state=55)

df_train.to_csv('spam_text_train.csv', index=False)
df_test.to_csv('spam_text_test.csv', index=False)

"""Getting 5% Seeds"""
train = df_train

spam = train[train['label'] == 1]
ham = train[train['label'] == 0]
print(ham.shape)
print(spam.shape)

spam_seed_05 = spam.sample(frac=.05, replace=False, random_state=79)
print(spam_seed_05.shape)
spam_seed_05.to_csv("spam_seed_05.csv", index=False)

ham_seed_05 = ham.sample(frac=.05, replace=False, random_state=79)
print(ham_seed_05.shape)
ham_seed_05.to_csv("ham_seed_05.csv", index=False)