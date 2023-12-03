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


def do_search(query):
   # do an actual search (grab from scraper, build df)
   # OLD BASELINE: results, wdf =  baseline_ranker(query, df)
   results, wdf = query_ranker(query, df,keyword_lines, np_embs)
   fig = px.scatter(wdf, x="rank", y="discordance",
	         size="discordance", color="country",
                 hover_name="title_en", size_max=60)
   fig.update_layout(
    autosize=False,
    width=500,
    height=800,
)
   data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   return results, data



@app.route('/results/q=<query>')
def results(query):
   results, plot = do_search(query)
   return render_template('result.html', query=query, results=results, plot=plot)


@app.route('/', methods=['GET','POST'])
def index():
   if request.method == 'GET':
      return render_template('index.html')
   else:
      query = request.form['query']
      return redirect(url_for('results',query=query))

if __name__ == '__main__':
   app.run(port=8000, debug=True)