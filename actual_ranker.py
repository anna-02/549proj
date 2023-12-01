import pandas as pd 

## Do our actual ranking
"""
Keywords, 
 >> word2vec for each of our keywords 
 >> then we cluster keywords ?  -- using dbscan 
 >> look @ makeup of clusters  -- differential makeup -> higher output 
    >> cluster size 


>> Discordance rating --> basically accumulate scores of all clusters for document 
    ( need to get  bidirectional keyword <--> cluster mapping)
    then do weighted sum of discordance relationship 

Discordance ratings for cluster: use signed element to allow 'directionality'
"""

class Clusterer(): 
    def __init__(self): 
        # load keywords by document 
        # then create a list of document_keywords by 'left join' with list of doc* keyworrds 
        