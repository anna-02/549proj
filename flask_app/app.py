from flask import Flask, render_template, redirect, url_for
import pandas as pd
from flask import request
from flask_functions import *
import json
import plotly.express as px
import plotly

app = Flask(__name__)
df = build_data(fname='../data/bag_of_words_translated-full_col_translated.csv', ftype='csv')


def do_search(query):
   results, wdf =  baseline_ranker(query, df)
   print(results)
   fig = px.scatter(wdf, x="rank", y="new rank",
	         size="discordance", color="country",
                 hover_name="title_en", log_x=True, size_max=60)
   data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   # fig.write_html("temp/plot.html")
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