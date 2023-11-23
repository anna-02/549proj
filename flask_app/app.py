from flask import Flask, render_template, redirect, url_for
from wtforms import Form, BooleanField, StringField, SubmitField, validators
import pandas as pd
from flask import request

app = Flask(__name__)

class BasicForm(Form):
    query = StringField("query",validators=[validators.DataRequired()])
    submit = SubmitField("Submit")

def get_offline_results():
   df = pd.read_csv('bag_of_words_translated - temp for figs.csv')
   return df


def do_search():
   results = ['I','am','tired']
   # iterate over each row in the subfeatures dataset
   # for subfeature_index, subfeature_row in subfeatures.iterrows():
   #   similarity = cosine_similarity(
   #       ast.literal_eval(subfeature_row['Description Embeddings']),
   #       embed_description)
    
   # # compute the cosine similarity between the query and all the rows in the corpus
   #   results.append({
   #       "score": similarity,
   #       "subfeature": subfeature_row['Name'],
   #       })

    
   #results = sorted(results, key=lambda x: x['score'], reverse=True)
   return results[:5]


@app.route('/results/q=<query>')
def results(query):
   results = do_search()
   return render_template('result.html', query=query, pages=results)


@app.route('/', methods=['GET','POST'])
def index():
   if request.method == 'GET':
      return render_template('index.html')
   else:
      query = request.form['query']
      return redirect(url_for('results',query=query))

if __name__ == '__main__':
   app.run(debug=True)