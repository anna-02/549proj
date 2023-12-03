from flask import Flask, render_template, redirect, url_for
import pandas as pd
from flask import request
from flask_functions import *
import json
import plotly.express as px
import plotly

df, np_embs, keyword_lines = build_data_new()
print('COLLL', df.columns)
query=None
results, wdf = query_ranker(query, df,keyword_lines, np_embs)
print(results)


