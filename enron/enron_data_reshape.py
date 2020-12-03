import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('enron/data/enron_original.csv')
df = df[['text', 'label']]
df['label'] = df['label'].replace({'ham': 0, 'spam': 1})

var1 = list(df["text"])
var = []
for i in var1:
    if len(i) > 500:
        var.append(i[:500])
    else:
        var.append(i)

length = []

for i in var1:
    length.append(len(i))

df["length"] = length
print(df["length"].describe())

length = sorted(length, reverse=True)
print(length[:50])

df['text'] = var

df = df[['text', 'label']]
#df.to_csv('enron/data/trunc_enron.csv', index=False)
#print(df.describe())

"""Taking a smaller samples of the full Enron emails dataset"""
df = pd.read_csv('enron/data/trunc_enron.csv', encoding='latin')

"""Test/Train split"""
df_train, df_test= train_test_split(df, test_size=0.33, random_state=55)

# df_train.to_csv('enron_email_train.csv', index=False)
# df_test.to_csv('enron_email_test.csv', index=False)

"""Getting 1% Seeds"""
spam = df_train[df_train['label'] == 1]
ham = df_train[df_train['label'] == 0]
print(ham.shape)
print(spam.shape)

spam_seed_05 = spam.sample(frac=.05, replace=False, random_state=79)
print(spam_seed_05.shape)
spam_seed_05.to_csv("enron/data/spam_seed_05.csv", index=False)

ham_seed_05 = ham.sample(frac=.05, replace=False, random_state=79)
print(ham_seed_05.shape)
ham_seed_05.to_csv("enron/data/ham_seed_05.csv", index=False)