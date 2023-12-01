import pandas as pd 
import numpy as np
import json
import ast

def load_df_with_keys(df_path:str,keys_col = 'keys')-> pd.DataFrame:
    if df_path[-4:] == '.csv': 
        df = pd.read_csv(df_path)
        if keys_col in df.columns and type(df[keys_col].iloc[0])  is str: 
            df[keys_col] = df[keys_col].apply(lambda st: ast.literal_eval(st)) 
    elif df_path[-5:] == '.json': 
        df = pd.read_json(df_path)
    else: 
        df = pd.read_pickle(df_path)
    return df.dropna(subset=['doc_en'])


def load_vecs(
        vecs_file = 'data/keyword_vecs.np.npy',
        lines_file = 'data/keyword_lines.json'
    )-> tuple[np.ndarray, dict[str,int]]:
    np_embs = np.load(vecs_file)
    with open(lines_file, "r") as file:
        keyword_lines = json.load(file)
        print(list(keyword_lines.keys())[0], type(list(keyword_lines.keys())[0]))
        first_k, first_v = next(iter(keyword_lines.items()))
        if type(first_k) is int or (type(first_k) is str and first_k.isnumeric()):
            keyword_lines = {keyword: int(i) for i, keyword in keyword_lines.items()}
        elif type(first_v) is str and first_v.is_numeric():
            keyword_lines = {keyword: int(i) for keyword, i in keyword_lines.items()}

    return np_embs, keyword_lines    
