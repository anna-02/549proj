import json
import os
import numpy as np
import pandas as pd
from utils import load_df_with_keys, load_vecs
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

KEEP_KEYS = ["keys", "country", "doc", "result_id", "title_en"]

def get_clusters(queries_df, word_dict, embeddings_np, method="hdbscan", **dbscan_args):
    # get keyword and keyword -> doc mapping (implicitly by order) x
    keywords_df = queries_df[KEEP_KEYS].explode("keys")
    keywords = keywords_df["keys"].to_list()

    # cluster based off of https://arxiv.org/pdf/2008.09470.pdf
    # default params:
    metric = lambda x, y: np.inner(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)) 

    if method.lower() == "dbscan":
        params = {'min_samples':7,"metric": metric}
        params.update(dbscan_args)

        clusterer = DBSCAN(**params)
    elif method.lower() == "hdbscan":
        params = {
            "min_cluster_size": 3,
            "metric": metric,
            "cluster_selection_method": "leaf",
        }

        # params = {
        #     "min_cluster_size": 3,
        #     'max_cluster_size': len(keywords_df)//6,
        #     "metric": metric,
        #     "cluster_selection_method": "eom",
        # }


        params.update(dbscan_args)

        clusterer = HDBSCAN(**params)
    else:
        raise ValueError("method must be in ['dbscan','hdbscan']")
    print(embeddings_np[word_dict[keywords[0]], :].shape)
    X = np.concatenate(
        [embeddings_np[word_dict[word] : word_dict[word] + 1, :] for word in keywords],
        axis=0,
    )
    print(X.shape)
    clusters = clusterer.fit_predict(X)
    print(clusters)
    keywords_df["cluster_id"] = clusters
    print(clusters.shape)

    return keywords_df

def get_doc_discordances(clusters_df:pd.DataFrame, agg_mode:str='directional',drop_nas=True) -> pd.DataFrame: 
    if agg_mode.lower() not in ['directional','absolute']: 
        raise ValueError(f"agg_mode must be in {'directional','absolute'}, was {agg_mode}")
    if drop_nas: 
        filt_df = lambda x: x[x['cluster_id'] != -1 ]
    else: 
        filt_df = lambda x:x
    cluster_stats = (
        filt_df(clusters_df).assign(
            cluster_size=1,
            discordance=filt_df(clusters_df)["country"].apply(
                lambda x: 1 if x == "us" else -1 if x == "ru" else 0
            ),
        )
        .groupby("cluster_id")
        .agg({"discordance": "mean", "cluster_size": "sum"})
    )
    cluster_df = clusters_df.join(cluster_stats, on='cluster_id')
    print(cluster_df)
    average_func = (lambda x: np.mean(np.abs(x))) if agg_mode == 'absolute' else np.mean
    aggby = {col:'first' for col in KEEP_KEYS }
    aggby['keys'] = list
    aggby['discordance'] = average_func
    aggby['cluster_size'] = 'mean'
    queries = cluster_df.groupby('result_id').agg(aggby)
    queries[['discordance','cluster_size']] = queries[['discordance','cluster_size']].fillna(0)
    queries['abs_discordance'] = np.abs(queries['discordance'].to_numpy())
    return queries.sort_values(['abs_discordance','cluster_size'],ascending=False), cluster_df, cluster_stats

    


if __name__ == "__main__":
    do_full_test = False
    test_one_query = True

    # load stuff from cached files
    test_df = load_df_with_keys("data/keywords_df2.csv")
    np_embs, keyword_lines = load_vecs(vecs_file='data/keyword_vecs.np.npy',lines_file='data/keyword_lines.json')

    if do_full_test: 
        # cluster 
        clusters_df = get_clusters(test_df, keyword_lines, np_embs,metric=lambda x, y: np.inner(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)) )
        # save our clusters for later use 
        clusters_df.to_csv("data/clusters_df.csv")
        print(clusters_df)
        print(clusters_df.value_counts("cluster_id"))
        print(clusters_df.value_counts("cluster_id").value_counts())
        # get some cursory info 
        cluster_infos = (
            clusters_df.assign(
                size=1,
                discordance=clusters_df["country"].apply(
                    lambda x: 1 if x == "us" else -1 if x == "ru" else 0
                ),
            )
            .groupby("cluster_id")
            .agg({"discordance": "mean", "size": "sum"})
        )
        print(cluster_infos)
        cluster_infos.to_csv("data/cluster_stats.csv")

        # process and rank clusters, get discordances, rank query results
        queries_ranking,_, cluster_stats = get_discordances(clusters_df) 
        print(queries_ranking)
        print(cluster_stats)

    
    if test_one_query: 
        # choose query at random 
        query = test_df.sample()['for_query_en'].iloc[0]
        print(f'testing on query:{query}')
        # take only our responses for that query
        query_df = test_df[test_df['for_query_en'] == query]
        # make clusters
        query_clusters = get_clusters(query_df,keyword_lines, np_embs)
        print(query_clusters)
        query_res_rankings,_,query_cluster_stats = get_discordances(query_clusters)
        print(query_cluster_stats)
        print(query_res_rankings)