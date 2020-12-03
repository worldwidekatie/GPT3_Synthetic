import pandas as pd
import openai

openai.api_key = ""


pos_ = pd.read_csv("imbd/data/pos_seed_05.csv")
pos_df = list(pos_['text'])

neg_ = pd.read_csv("imbd/data/neg_seed_05.csv")
neg_df = list(neg_['text'])

def gen_data(df_column, sentiment, shots, engine="ada"):
    data = []
    for i in range(len(df_column)):
        if shots == 1:
            prompt = f"The following are movie reviews with a {sentiment} sentiment. REVIEW: {df_column[i]} REVIEW:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="REVIEW:")
            for i in response['choices']:
                data.append(i['text'])
        if shots == 2:
            prompt = f"The following are movie reviews with a {sentiment} sentiment. REVIEW: {df_column[i]} REVIEW: {df_column[i-1]} REVIEW:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="REVIEW:")
            for i in response['choices']:
                data.append(i['text'])
        
        if shots == 3:
            prompt = f"The following are movie reviews with a {sentiment} sentiment. REVIEW: {df_column[i]} REVIEW: {df_column[i-1]} REVIEW: {df_column[i-2]} REVIEW:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="REVIEW:")
            for i in response['choices']:
                data.append(i['text'])        
        
        if shots == 4:
            prompt = f"The following are movie reviews with a {sentiment} sentiment. REVIEW: {df_column[i]} REVIEW: {df_column[i-1]} REVIEW: {df_column[i-2]} REVIEW: {df_column[i-3]} REVIEW:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="REVIEW:")
            for i in response['choices']:
                data.append(i['text'])

        if shots == 5:
            prompt = f"The following are movie reviews with a {sentiment} sentiment. REVIEW: {df_column[i]} REVIEW: {df_column[i-1]} REVIEW: {df_column[i-2]} REVIEW: {df_column[i-3]} REVIEW: {df_column[i-4]} REVIEW:"
            response = openai.Completion.create(engine=engine, prompt=prompt, 
                                        max_tokens=75, temperature=.8, top_p=1, n=20, 
                                        stream=False, stop="REVIEW:")
            for i in response['choices']:
                data.append(i['text'])
    return data

# One Shot Ada
pos = pd.DataFrame(gen_data(pos_df, 'positive', 1), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/1shot_pos_05_ada.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 1), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/1shot_neg_05_ada.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/1shot_05_train_ada.csv')

# One Shot Davinci
pos = pd.DataFrame(gen_data(pos_df, 'positive', 1, engine='davinci'), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/1shot_pos_05_davinci.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 1, engine='davinci'), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/1shot_neg_05_davinci.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/1shot_05_train_davinci.csv')


# Two Shot Ada
pos = pd.DataFrame(gen_data(pos_df, 'positive', 2), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/2shot_pos_05_ada.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 2), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/2shot_neg_05_ada.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/2shot_05_train_ada.csv')

# Two Shot Davinci
pos = pd.DataFrame(gen_data(pos_df, 'positive', 2, engine='davinci'), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/2shot_pos_05_davinci.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 2, engine='davinci'), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/2shot_neg_05_davinci.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/2shot_05_train_davinci.csv')

# Three Shot Ada
pos = pd.DataFrame(gen_data(pos_df, 'positive', 3), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/3shot_pos_05_ada.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 3), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/3shot_neg_05_ada.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/3shot_05_train_ada.csv')

# Three Shot Davinci
pos = pd.DataFrame(gen_data(pos_df, 'positive', 3, engine='davinci'), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/3shot_pos_05_davinci.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 3, engine='davinci'), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/3shot_neg_05_davinci.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/3shot_05_train_davinci.csv')

# Four Shot Ada
pos = pd.DataFrame(gen_data(pos_df, 'positive', 4), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/4shot_pos_05_ada.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 4), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/4shot_neg_05_ada.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/4shot_05_train_ada.csv')

# Four Shot Davinci
pos = pd.DataFrame(gen_data(pos_df, 'positive', 4, engine='davinci'), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/4shot_pos_05_davinci.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 4, engine='davinci'), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/4shot_neg_05_davinci.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/4shot_05_train_davinci.csv')

# Five Shot Ada
pos = pd.DataFrame(gen_data(pos_df, 'positive', 5), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/5shot_pos_05_ada.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 5), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/5shot_neg_05_ada.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/5shot_05_train_ada.csv')

# Five Shot Davinci
pos = pd.DataFrame(gen_data(pos_df, 'positive', 5, engine='davinci'), columns=["text"])
pos['label'] = 1
pos.to_csv("imbd/data/5shot_pos_05_davinci.csv", index=False)

neg = pd.DataFrame(gen_data(neg_df, 'negative', 5, engine='davinci'), columns=["text"])
neg['label'] = 0
neg.to_csv("imbd/data/5shot_neg_05_davinci.csv", index=False)

df = pd.concat([pos, neg])
df.to_csv('imbd/data/5shot_05_train_davinci.csv')