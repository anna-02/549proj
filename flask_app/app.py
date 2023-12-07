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

def do_search(query):
   # do an actual search (grab from scraper, build df)
   # OLD BASELINE: results, wdf =  baseline_ranker(query, df)
   results, wdf, cluster_df = query_ranker(query, df,keyword_lines, np_embs,eps=0.05,min_samples=3)
   cluster_df['cluster_prop'] = np.abs(cluster_df['discordance'].to_numpy()) 
   wdf = wdf.drop(columns=['keys','cluster_id','cluster_prop'])
   x = cluster_df[['cluster_prop','cluster_id','keys','result_id']]
   gdf = wdf.set_index('result_id').join(x.set_index('result_id')).reset_index()
   gdf['discordance_log'] = np.abs(np.log(gdf['discordance']).to_numpy()*10)
   gdf['cluster_agg'] =  gdf['cluster_y'] # gdf['cluster_x'] +
   fig = px.scatter(gdf, x="cluster_agg", y="result_id",
	         size="discordance_log", color="cluster_id",
                 hover_name="keys", opacity=0.8, size_max=30)
   fig.update_layout(
    autosize=False,
    width=500,
    height=600,
)
   data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   return results, data



@app.route('/results/q=<query>')
def results(query):
   results, plot = do_search(query)
   ru_query = df[df['for_query_en'] == query]['for_query'].iloc[0]
   return render_template('result.html', query=query,ru_query=ru_query, results=results, plot=plot)


@app.route('/', methods=['GET','POST'])
def index(queries=queries):

   if request.method == 'GET':
      return render_template('index.html',queries=queries)
   else:
      query = request.form['query']
      return redirect(url_for('results',query=query))

if __name__ == '__main__':
   app.run(port=8000, debug=True)