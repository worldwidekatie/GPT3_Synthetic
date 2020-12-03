import pandas as pd
import openai

openai.api_key = ""


spam_ = pd.read_csv("sms/data/spam_seed_05.csv")
spam_df = list(spam_['text'])

ham_ = pd.read_csv("sms/data/ham_seed_05.csv")
ham_df = list(ham_['text'])

def gen_data(df_column, label, shots, engine="ada"):
    data = []
    for i in range(len(df_column)):
        if shots == 1:
            prompt = f"TEXT MESSAGE: {df_column[i]} TEXT MESSAGE:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=50, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="TEXT MESSAGE:")
            for i in response['choices']:
                data.append(i['text'])
        if shots == 2:
            prompt = f"TEXT MESSAGE: {df_column[i]} TEXT MESSAGE: {df_column[i-1]} TEXT MESSAGE:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=50, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="TEXT MESSAGE:")
            for i in response['choices']:
                data.append(i['text'])
        
        if shots == 3:
            prompt = f"TEXT MESSAGE: {df_column[i]} TEXT MESSAGE: {df_column[i-1]} TEXT MESSAGE: {df_column[i-2]} TEXT MESSAGE:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=50, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="TEXT MESSAGE:")
            for i in response['choices']:
                data.append(i['text'])        
        
        if shots == 4:
            prompt = f"TEXT MESSAGE: {df_column[i]} TEXT MESSAGE: {df_column[i-1]} TEXT MESSAGE: {df_column[i-2]} TEXT MESSAGE: {df_column[i-3]} TEXT MESSAGE:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=50, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="TEXT MESSAGE:")
            for i in response['choices']:
                data.append(i['text'])        
        
        if shots == 5:
            prompt = f"TEXT MESSAGE: {df_column[i]} TEXT MESSAGE: {df_column[i-1]} TEXT MESSAGE: {df_column[i-2]} TEXT MESSAGE: {df_column[i-3]} TEXT MESSAGE: {df_column[i-4]} TEXT MESSAGE:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=50, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="TEXT MESSAGE:")
            for i in response['choices']:
                data.append(i['text'])
    return data

# One Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 1), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/1shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 1), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/1shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/1shot_05_train_ada.csv')

# One Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 1, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/1shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 1, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/1shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/1shot_05_train_davinci.csv')


# Two Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 2), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/2shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 2), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/2shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/2shot_05_train_ada.csv')

# Two Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 2, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/2shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 2, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/2shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/2shot_05_train_davinci.csv')

# Three Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 3), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/3shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 3), columns=["text"])
ham['label'] = 0
ham.to_csv("3sms/data/shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/3shot_05_train_ada.csv')

# Three Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 3, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/3shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 3, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/3shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/3shot_05_train_davinci.csv')

# Four Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 4), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/4shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 4), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/4shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/4shot_05_train_ada.csv')

# Four Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 4, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/4shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 4, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/4shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/4shot_05_train_davinci.csv')

# Five Shot Ada
spam = pd.DataFrame(gen_data(spam_df, 'spam', 5), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/5shot_spam_05_ada.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 5), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/5shot_ham_05_ada.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/5shot_05_train_ada.csv')

# Five Shot Davinci
spam = pd.DataFrame(gen_data(spam_df, 'spam', 5, engine='davinci'), columns=["text"])
spam['label'] = 1
spam.to_csv("sms/data/5shot_spam_05_davinci.csv", index=False)

ham = pd.DataFrame(gen_data(ham_df, 'normal', 5, engine='davinci'), columns=["text"])
ham['label'] = 0
ham.to_csv("sms/data/5shot_ham_05_davinci.csv", index=False)

df = pd.concat([spam, ham])
df.to_csv('sms/data/5shot_05_train_davinci.csv')