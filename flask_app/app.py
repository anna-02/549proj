from flask import Flask, render_template, redirect, url_for
import pandas as pd
from flask import request
from flask_functions import *
import json
import plotly.express as px
import plotly

app = Flask(__name__)
df_old = build_data(fname='../data/bag_of_words_translated-full_col_translated.csv', ftype='csv')
df, np_embs, keyword_lines= build_data_new()
queries = set(df['for_query_en'])

def do_search(query,eps=0.3, min_samples=3,country_bias=False):
   # do an actual search (grab from scraper, build df)
   # OLD BASELINE: results, wdf =  baseline_ranker(query, df)
   results, wdf, cluster_df = query_ranker(query, df,keyword_lines, np_embs,eps=eps,min_samples=min_samples,bias=country_bias)
   print('*******',eps, min_samples,country_bias)
   cluster_df['cluster_prop'] = np.abs(cluster_df['discordance'].to_numpy()) 
   wdf = wdf.drop(columns=['keys','cluster_id','cluster_prop'])
   x = cluster_df[['cluster_prop','cluster_id','keys','result_id']]
   gdf = wdf.set_index('result_id').join(x.set_index('result_id')).reset_index()
   gdf['discordance_exp'] = np.exp(gdf['discordance'].to_numpy()*10 )# np.abs(np.log(gdf['discordance']).to_numpy()*10)
   gdf['cluster_agg'] =  gdf['cluster_y'] # gdf['cluster_x'] +
   fig = px.scatter(gdf, x="discordance",y="cluster_agg",
	         size="discordance_exp", color="cluster_id",
                 hover_data=['keys', 'country', 'title_en'] , opacity=0.8, size_max=30)
   fig.update_layout(
    autosize=False,
    width=500,
    height=600,
)
   data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   return results, data



@app.route('/results/q=<query>',methods=['GET','POST'])
def results(query):
   if request.method == 'POST':
      eps = float(request.form['eps']) if request.form['eps'] != '' else 0.5
      min_samples = int(request.form['min_samples']) if request.form['min_samples'] != '' else 3
      country_bias= request.form['country_bias'] if request.form['country_bias'] != 'None' else False
      results, plot = do_search(query,eps=eps,min_samples=min_samples,country_bias=country_bias)
   else: 
      eps = 0.5
      min_samples = 3
      country_bias='None'
      results, plot = do_search(query,eps=eps,min_samples=min_samples)

   ru_query = df[df['for_query_en'] == query]['for_query'].iloc[0]
   return render_template('result.html', query=query,ru_query=ru_query, results=results, plot=plot, eps=eps,min_samples=min_samples,country_bias=country_bias)


@app.route('/', methods=['GET','POST'])
def index(queries=queries):

   if request.method == 'GET':
      return render_template('index.html',queries=queries)
   else:
      query = request.form['query']
      return redirect(url_for('results',query=query))

if __name__ == '__main__':
   app.run(port=8000, debug=True)