# preprocess.py - example helpers
import pandas as pd
def load_tumor_volumes(path): 
    df = pd.read_csv(path)
    df['Day'] = df['Day'].astype(int)
    return df

def baseline_normalize(df):
    # compute percent change vs day0 for each Model
    out = []
    for model, g in df.groupby('Model'):
        base = g.loc[g['Day']==0,'Volume_mm3'].values[0]
        g = g.copy()
        g['PctChange'] = (g['Volume_mm3']/base - 1) * 100
        out.append(g)
    return pd.concat(out, ignore_index=True)

