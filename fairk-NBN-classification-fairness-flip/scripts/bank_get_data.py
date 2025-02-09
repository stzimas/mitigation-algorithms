import pandas as pd
#https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

df = pd.read_csv("../data/bank.csv")
df['marital'] = df['marital'].where(df['marital'] == 'married', 'non-married')
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 2})
df.rename(columns={"deposit": "class"}, inplace=True)
df.to_csv("../data/bank_wtarget.csv")
