from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def build_data(fname ='../data/bag_of_words_translated.xlsx',sheet_name='full_col_translated',ftype='xlsx'):
    if ftype == 'csv':
        ann_df = pd.read_csv(fname)
    else:
        ann_df = pd.read_excel(fname,sheet_name=sheet_name)
    ann_df = ann_df.dropna(subset=['for_query','title','rank','links','country','discordance'])
    ann_df = ann_df.reset_index(drop=True).reset_index()
    ann_df['bow'] = ann_df.title_en.astype(str) + ann_df.snippet_en.astype(str)  +ann_df.doc_en.astype(str) 
    return ann_df
    #queries_ann = set(ann_df['for_query'])
    #ann_df[['index','for_query','for_query_en','title','rank','country','discordance']]

def format_row_val(rv, chars=100):
    if len(str(rv)) < 5:
        return ''
    return str(rv)[:chars]

def baseline_ranker(q, x, web=True):
    '''where q is a query and x is a df with "bow" bag of words col set as cocatenation of title/snippet/doc'''
    x = x.copy()
    us = x[(x['for_query']==q) & (x['country']=='us')]
    ru = x[(x['for_query']==q) & (x['country']=='ru')]
    idx_scores = []
    web_results = []
    # wr = {'title':'','document':'','link':'','text':'','discordance':''}
    # print('for query:', q)
    for i,row in us.iterrows():
        toks = ' '.join(row['bow']).split(' ') #(' '.join(row['title_en']) +  ' '.join(row['snippet_en']) + ' '.join(row['doc_en'])).split(' ') 
        tok_bucket = ' '.join(ru['bow']).split(' ') #(' '.join(ru['title_en']) +  ' '.join(ru['snippet_en']) + ' '.join(ru['doc_en'])).split(' ') 
        no_overlap = set(toks) - set(tok_bucket)
        idx_scores.append((row['index'], 'us', row['title_en'][:25],(len(no_overlap) / len(toks)) * 5))
        row['snippet'] = format_row_val(row['snippet'])
        row['doc'] = format_row_val(row['doc'])
        row['discordance'] = len(no_overlap) / len(toks) * 5
        web_results.append(row)
        
    for i,row in ru.iterrows():
        toks = ' '.join(row['bow']).split(' ') #(' '.join(row['title_en']) +  ' '.join(row['snippet_en']) + ' '.join(row['doc_en'])).split(' ') 
        tok_bucket = ' '.join(us['bow']).split(' ')#(' '.join(us['title_en']) +  ' '.join(us['snippet_en']) + ' '.join(us['doc_en'])).split(' ') 
        no_overlap = set(toks) - set(tok_bucket)
        row['snippet'] = format_row_val(row['snippet'])
        row['doc'] = format_row_val(row['doc'])
        row['discordance'] = len(no_overlap) / len(toks) * 5
        web_results.append(row)
        idx_scores.append((row['index'], 'ru',row['title_en'][:25], (len(no_overlap) / len(toks)) * 5))

    return sorted(web_results, key=lambda x: x['discordance'], reverse=True)

def run_stats(ann_df):
    queries_ann = set(ann_df['for_query'])
    base_rec = []
    stat_rec = []
    for q in list(queries_ann):
        # baseline DOES NOT ACTUALLY USE DISC SCORES FROM ANN DF
        baseline = baseline_ranker(q, ann_df)
        # print(baseline)

        truth = ann_df[ann_df['for_query']==q]['discordance'].sort_values(ascending=False).to_numpy() # when 3 or above
        query_en = ann_df[ann_df['for_query']==q]['for_query_en'].iloc[0]
        mp = average_precision_score([1 if t > 2 else 0 for t in truth ], [1 if b[3] > 2 else 0 for b in baseline ])
        

        baseline.sort(key=lambda x:x[-1])
        # print(baseline)
        #  baseline_scores = -np.sort(-baseline_scores)

        baseline_scores = np.array([ann_df.loc[b[0],'discordance'] for b in baseline]) #* 5.0
        # print(baseline_scores.shape, len(truth))
        # print(baseline_scores)
        # print(truth)
        ng = ndcg_score(truth[:,None].T, baseline_scores[:,None].T ,k=10)
        # print(ng)
        base_rec.append([q, 'map',mp, query_en])
        base_rec.append([q, 'ndcg',ng, query_en])


        stat_rec.append([q,query_en,stats.ks_2samp(truth, [b[3] for b in baseline])[1]])
    statdf=  pd.DataFrame().from_records(stat_rec, columns=['query','query_en','value'])
    basedf = pd.DataFrame().from_records(base_rec, columns=['query','score','value','query_en'])