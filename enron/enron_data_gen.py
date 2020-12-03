import pandas as pd
import openai

openai.api_key = ""


spam_ = pd.read_csv("enron/data/spam_seed_05.csv")
spam_df = list(spam_['text'])

ham_ = pd.read_csv("enron/data/ham_seed_05.csv")
ham_df = list(ham_['text'])

def gen_data(df_column, label, shots, engine="ada"):
    data = []
    for i in range(len(df_column)):
        if shots == 1:
            prompt = f"The following emails are {label} emails sent to Enron employees. {df_column[i]} Subject:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="Subject:")
            for i in response['choices']:
                data.append(i['text'])
        
        if shots == 2:
            prompt = f"The following emails are {label} emails sent to Enron employees. {df_column[i]} {df_column[i-1]} Subject:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="Subject:")
            for i in response['choices']:
                data.append(i['text'])
        
        if shots == 3:
            prompt = f"The following emails are {label} emails sent to Enron employees. {df_column[i]} {df_column[i-1]} {df_column[i-2]} Subject:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="Subject:")
            for i in response['choices']:
                data.append(i['text'])

        if shots == 4:
            prompt = f"The following emails are {label} emails sent to Enron employees. {df_column[i]} {df_column[i-1]} {df_column[i-2]} {df_column[i-3]} Subject:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="Subject:")
            for i in response['choices']:
                data.append(i['text'])

        if shots == 5:
            prompt = f"The following emails are {label} emails sent to Enron employees. {df_column[i]} {df_column[i-1]} {df_column[i-2]} {df_column[i-3]} {df_column[i-4]}Subject:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="Subject:")
            for i in response['choices']:
                data.append(i['text'])
    return data

# One Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 1), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/1shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 1), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/1shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/1shot_05_train_ada.csv')

# One Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 1, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/1shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 1, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/1shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/1shot_05_train_davinci.csv')


# Two Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 2), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/2shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 2), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/2shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/2shot_05_train_ada.csv')

# Two Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 2, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/2shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 2, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/2shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/2shot_05_train_davinci.csv')

# Three Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 3), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/3shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 3), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/3shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/3shot_05_train_ada.csv')

# Three Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 3, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/3shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 3, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/3shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/3shot_05_train_davinci.csv')

# Four Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 4), columns=["text"])
spam['label'] = 1
spam.to_csv("4enron/data/shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 4), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/4shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/4shot_05_train_ada.csv')

# Four Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 4, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/4shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 4, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/4shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/4shot_05_train_davinci.csv')

# Five Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 5), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/5shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 5), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/5shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/5shot_05_train_ada.csv')

# Five Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 5, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("enron/data/5shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 5, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("enron/data/5shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('enron/data/5shot_05_train_davinci.csv')