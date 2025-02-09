import pandas as pd
#https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
df = pd.read_csv("../data/UCI_Credit_Card.csv")
df = df.drop(columns=["ID"])
#SEX: Gender (1=male, 2=female)
df.rename(columns={"SEX": "Sex",
                   "default.payment.next.month":"class"}, inplace=True)
df["Sex"] = df["Sex"].replace({1: "male", 2: "female"})
df["class"] = df["class"].replace({1: 1, 0: 2})

df.to_csv("../data/uci_cc_wtarget.csv")

k=1
