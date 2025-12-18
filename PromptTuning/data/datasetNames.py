import pandas as pd

file='sample_pairs.csv'
with open(file):
    df=pd.read_csv(file)
#print(df.head)
dsNames=df['dataset']
print(dsNames.unique())