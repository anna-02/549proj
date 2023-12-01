import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm
import requests
import json



def build_date_ranges(num_years,start_year = 2022): 
    time_frames = []
    for i in range(num_years*12):
        month = i%12+1
        year = start_year + (i // 12)
        nex_month = (i+1)%12+1
        next_year = start_year + ((i+1)// 12)
        time_frames.append(f'{year}-{month:02d}-01 {next_year:04d}-{nex_month:02d}-01')
    return time_frames

def get_trend(timeframe, reigon='US',): 
    pytrend = TrendReq()
    # for timeframe in timeframes: 
    pytrend.build_payload(kw_list=[''],geo=reigon,timeframe=timeframe)
    res = pytrend.related_queries()
    top = res['']['top']
    top['cat'] = 'top'
    rising = res['']['rising']
    rising['cat'] = 'rising'

    return pd.concat([top,rising]).reset_index(names='rank')

def get_trendings(timeframes,reigon='US'): 
    dfs = []
    for tf in tqdm(timeframes): 
        df = get_trend(tf,reigon=reigon)
        df['timeframe'] = tf
        dfs.append(df)
    return pd.concat(dfs)


def google_search(search_term,reigon='us', **kwargs):
    api_key = CSE_API_KEY
    cse_id = CSE_ID
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': search_term,
        'key': api_key,
        'cx': cse_id,
        'gl':reigon,
        **kwargs
    }
    response = requests.get(url, params=params)
    return response.json()

category_dict = {
    'People & Society': 14,
    'Ethnic & Identity Groups':56,
    'Discrimination & Identity Relations':1205,
    'Politics':396,
    'News':16,
    'World News':1209,
    'Business News':784,
}




if __name__ == "__main__":
    with open('GOOG_API_KEY','r') as file:
        CSE_API_KEY = file.read().strip()
    CSE_ID = '43624319dd4de4487'

    print(json.dumps(google_search('usps tracking',reigon='us'),indent=4))
    print(json.dumps(google_search('usps tracking',reigon='ru'),indent=4))


    timeframes = build_date_ranges(2,2020)
    print(timeframes)
    trends = get_trendings(timeframes=timeframes)
    print(trends)
    trends.to_csv('data/english_trending_queries.csv')
    top_queries_filtered = trends.drop_duplicates(subset='query')

