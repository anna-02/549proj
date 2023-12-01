import json
import os
import numpy as np
import pandas as pd
from utils import load_df_with_keys, load_vecs
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


def get_clusters(queries_df, word_dict, embeddings_np, method="hdbscan", **dbscan_args):
    # get keyword and keyword -> doc mapping (implicitly by order) x
    keywords_df = queries_df[
        ["keys", "country", "doc", "result_id", "title_en"]
    ].explode("keys")
    keywords = keywords_df["keys"].to_list()

    # cluster based off of https://arxiv.org/pdf/2008.09470.pdf
    # default params:

    if method.lower() == "dbscan":
        params = {'min_samples':7,"metric": "euclidean"}
        params.update(dbscan_args)

        clusterer = DBSCAN(**params)
    elif method.lower() == "hdbscan":
        params = {
            "min_cluster_size": 7,
            "metric": "euclidean",
            "cluster_selection_method": "leaf",
        }

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


if __name__ == "__main__":
    test_df = load_df_with_keys("data/keywords_df2.csv")
    np_embs, keyword_lines = load_vecs(vecs_file='data/keyword_vecs.np.npy',lines_file='data/keyword_lines.json')


    clusters_df = get_clusters(test_df, keyword_lines, np_embs,metric=lambda x, y: np.inner(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)) )
    clusters_df.to_csv("data/clusters_df.csv")
    print(clusters_df)
    print(clusters_df.value_counts("cluster_id"))
    print(clusters_df.value_counts("cluster_id").value_counts())
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
