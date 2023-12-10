from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd
from utils import load_df_with_keys, load_vecs
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


KEEP_KEYS = ["keys", "country", "result_id", "title_en","snippet_en"]
KEYS_FOR_RES = ["result_id", "country",'for_query','for_query_en', "title_en", "snippet_en","doc_en","rank","links",'annotated_discordance']


def build_data(fname ='data/bag_of_words_translated.xlsx',sheet_name='full_col_translated',ftype='xlsx'):
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

def build_data_new(fname='data/keywords_df2.csv', vecs_file='data/keyword_vecs.np.npy', lines_file='data/keyword_lines.json'):
    test_df = load_df_with_keys(fname)
    np_embs, keyword_lines = load_vecs(vecs_file=vecs_file,lines_file=lines_file)
    test_df['doc_en'] = test_df['doc_en'].apply(lambda x: x.replace('\t','').replace('\n',''))
    test_df = test_df.rename(columns={'discordance':'annotated_discordance'})#['annotated_discordance'] = test_df['discordance']
    test_df = test_df.dropna(subset='annotated_discordance')
    return test_df, np_embs, keyword_lines


def format_row_val(rv, chars=100):
    if len(str(rv)) < 5:
        return ''
    return str(rv)[:chars]



def get_clusters(queries_df, word_dict, embeddings_np, method="dbscan", **dbscan_args):
    '''Returns df with columns corresponding to cluster id keys(keywords),country, docid, title'''
    # get keyword and keyword -> doc mapping (implicitly by order) x
    keywords_df = queries_df[KEEP_KEYS].explode("keys")
    keywords = keywords_df["keys"].to_list()
    # cluster based off of https://arxiv.org/pdf/2008.09470.pdf
    # default params:
    metric = lambda x, y: np.inner(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)) 

    if method.lower() == "dbscan":
        params = {'min_samples':3,'eps':0.5}
        print(dbscan_args)
        params.update(dbscan_args)
        clusterer = DBSCAN(**params)
    elif method.lower() == "hdbscan":
        params = {
            "min_cluster_size": 3,
            "metric": metric,
            "cluster_selection_method": "leaf",
        }
        params.update(dbscan_args)
        clusterer = HDBSCAN(**params)
    else:
        raise ValueError("method must be in ['dbscan','hdbscan']")
    X = np.concatenate(
        [embeddings_np[word_dict[word] : word_dict[word] + 1, :] for word in keywords],
        axis=0,
    )
    clusters = clusterer.fit_predict(X)
    keywords_df["cluster_id"] = clusters
    axes = [ X[i].reshape(-1,2).sum(axis=0) for i in range(len(X))]
    keywords_df['cluster_x'] = [i[0] for i in axes] # X
    keywords_df['cluster_y'] = [i[1] for i in axes]# a

    return keywords_df

def get_cluster_scores(clusters_df:pd.DataFrame, agg_mode:str='directional',bias=False, drop_nas=True) -> pd.DataFrame: 
    '''Cluster and get proportions of any one country, if bias is `us` or `ru`
        make that country 1 and the other 0 so mean is calculated for that country only
        and its results are less discordant'''
    if agg_mode.lower() not in ['directional','absolute']: 
        raise ValueError(f"agg_mode must be in {'directional','absolute'}, was {agg_mode}")
    if drop_nas: 
        filt_df = lambda x: x[x['cluster_id'] != -1 ]
    else: 
        filt_df = lambda x:x
    us_val, ru_val = 1, -1
    if bias and bias != 'None':
        us_val = 1 if bias == 'us' else 0
        ru_val = 1 if bias == 'ru' else 0 
    cluster_stats = (
        filt_df(clusters_df).assign(
            cluster_size=1,
            discordance=filt_df(clusters_df)["country"].apply(
                lambda x: us_val if x == "us" else ru_val if x == "ru" else 0
            ),
        )
        .groupby("cluster_id")
        .agg({"discordance": "mean", "cluster_size": "sum"})
    )
    cluster_df = clusters_df.join(cluster_stats, on='cluster_id')
    average_func = (lambda x: np.mean(np.abs(x))) if agg_mode == 'absolute' else np.mean
    aggby = {col:'first' for col in KEEP_KEYS }
    aggby['keys'] = list
    aggby['discordance'] = average_func
    aggby['cluster_size'] = 'mean'
    docs = cluster_df.groupby('result_id').agg(aggby)
    docs[['discordance','cluster_size']] = docs[['discordance','cluster_size']].fillna(0)
    docs['abs_discordance'] = np.abs(docs['discordance'].to_numpy())
    cluster_df['abs_discordance'] = np.abs(cluster_df['discordance'].to_numpy())
    return docs.sort_values(['abs_discordance','cluster_size'],ascending=False), cluster_df, cluster_stats



def geo_prop(g):
    p = max(len(g[g['country']=='us']), len(g[g['country']=='ru']))
    return p/len(g)

def get_group_discordance(g):
    '''Compute results discordance from calculated cluster discordances'''
    # for all clustered key words (if none are, return 1 as max discordance)
    # compute (all keywords * weights )divided by (keyword num and weights sum)
    # (k1*w1) + (k2*w2 + k3*w2) + (k4*w3) / (k1+k2+k3+k4)(w1+w2+w3)
    nn = g[g['cluster_id']!=-1]
    # if all noise, max discordance
    if len(nn) <= 0:
        return 1
    # else, calculate the average cluster discordance weighted by the proportion
    # of keywords in clusters
    return 1 - (nn['cluster_prop'].mean() * len(nn)/len(g))


def get_doc_discordances(clusters_df:pd.DataFrame) -> pd.DataFrame: 
    clusters_df['cluster_prop'] = clusters_df['abs_discordance'].to_numpy()
    clusters_df = clusters_df.reset_index(drop=True)

    gd = clusters_df.groupby('result_id').apply(get_group_discordance)
    docs_df = clusters_df.groupby('result_id').first().reset_index()
    docs_df['discordance'] = docs_df['result_id'].map(dict(gd))
    return docs_df.sort_values(['discordance'],ascending=False)


def query_ranker(query, test_df, keyword_lines, np_embs, eps=0.5,min_samples=3,bias=False):
    if query is None:
        # choose query at random 
        query = test_df.sample()['for_query_en'].iloc[0]
    print(f'testing on query:{query}')
    # take only our responses for that query
    query_df = test_df[test_df['for_query_en'] == query]
    # make clusters, get discordance scores, 
    query_clusters = get_clusters(query_df,keyword_lines, np_embs, method='dbscan', eps=eps, min_samples=min_samples)
    docs, cluster_df, cluster_stats = get_cluster_scores(query_clusters,drop_nas=False,bias=bias)
    cdf = cluster_df.copy()
    df = get_doc_discordances(cluster_df)
    ret_df=  query_df[KEYS_FOR_RES].merge(df).sort_values(by='discordance',ascending=False)
    ret_wr = format_df(ret_df)
    return ret_wr, ret_df, cdf


def format_df(df, chars=100):
    wr = []
    for i,row in df.iterrows():
        row['snippet_en'] = format_row_val(row['snippet_en'])
        row['doc_en'] = format_row_val(row['doc_en'])
        wr.append(row)
    return wr


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
        idx_scores.append([row['rank'], 'us', row['title_en'][:25],(len(no_overlap) / len(toks)) * 5,0])
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
        idx_scores.append([row['rank'], 'ru',row['title_en'][:25], (len(no_overlap) / len(toks)) * 5, 0])
    sorted_idx = sorted(idx_scores, key=lambda x: x[3], reverse=True)
    for i, x in enumerate(sorted_idx):
        sorted_idx[i][-1] = i
    sorted_wr= sorted(web_results, key=lambda x: x['discordance'], reverse=True)
    wf = pd.DataFrame().from_records(idx_scores, columns=['rank','country','title_en','discordance','new rank'])
    return sorted_wr, wf

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