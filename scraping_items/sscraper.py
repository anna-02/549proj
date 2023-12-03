from bs4 import BeautifulSoup
import requests, json, lxml
import pandas as pd
import time
from random import randint


COUNTRY = 'ru'
LANGUAGE = 'ru'

df = pd.read_csv('queries.csv')
read_df = pd.read_csv('google_scrape_results.csv')
queries = list(df['query'])
data = []
user_agent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1',
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18363',
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8",
    "Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36"
]


# SCRAPEOPS_API_KEY = '8b7a1986-abc5-4ce1-8fe5-a360ece2f838'

# def get_headers_list():
#   response = requests.get('http://headers.scrapeops.io/v1/browser-headers?api_key=' + SCRAPEOPS_API_KEY)
#   json_response = response.json()
#   return json_response.get('result', [])

for q in queries:
   
    grab_num = 20
    curr_count = 0
    if q in set(read_df['for_query']):
        curr_count = len(read_df[(read_df['for_query']==q) & (read_df['country']==COUNTRY)].drop_duplicates())
    if curr_count > 20:
        print('skipping', q, 'with', curr_count, 'in data already')
        continue
    grab_num = 20 - curr_count
    print('trying query...', q, 'for', grab_num)
    # https://docs.python-requests.org/en/master/user/quickstart/#passing-parameters-in-urls
    params = {
        "q": q,         # query example
        "hl": LANGUAGE,          # language
        "gl": COUNTRY,          # country of the search, UK -> United Kingdom
        "start": 0,          # number page by default up to 0
        "num": grab_num          # parameter defines the maximum number of results to return.
    }
    
    # header_list = get_headers_list()
    headers = {'User-Agent': user_agent_list[randint(0,len(user_agent_list)-1)]}
    page_num = 0

    # while page_num < 1:
    #     page_num += 1
    #     print(f"page: {page_num}")
    time.sleep(randint(30,40))
    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
    
    if html.status_code == 404 or html.status_code == 429:
        with open('google_results_up_to_{0}.jsonl'.format(q), 'wt') as f:
            print('failed at query', q)
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
            print(html.text)

            print('________writing past data________')
            time.sleep(100)
            html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
        
    soup = BeautifulSoup(html.text, 'lxml')
    rank = 1
    for result in soup.select(".tF2Cxc"): # .serp-item.serp-item_card 
        title = result.select_one(".DKV0Md").text
        try:
            snippet = result.select_one(".lEBKkf span").text # OrganicTitleContentSpan.organic__title
        except:
            snippet = None
        links = result.select_one(".yuRUbf a")["href"] # Link.Link_theme_normal.OrganicTitle-Link.organic__url.link
        rank += 1
        data.append({
        "for_query":q,
        "title": title,
        "snippet": snippet,
        'rank': rank,
        "links": links,
        "country": COUNTRY
        })
    rank = 1
    
    # if soup.select_one(".d6cvqb a[id=pnnext]"):
    #     params["start"] += 10
    # else:
    #     print(html.text)
    #     break

with open('google_results_fin_up_to_{0}.jsonl'.format(data[-1]['for_query']), 'wt') as f:
    f.write(json.dumps(data, indent=2, ensure_ascii=False))