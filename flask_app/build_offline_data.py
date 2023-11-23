import numpy as np
import pandas as pd

# connection = sqlite3.connect('database.db')


# with open('schema.sql') as f:
#     connection.executescript(f.read())

# cur = connection.cursor()

# cur.execute("INSERT INTO posts (title, content) VALUES (?, ?)",
#             ('First Post', 'Content for the first post')
#             )

# cur.execute("INSERT INTO posts (title, content) VALUES (?, ?)",
#             ('Second Post', 'Content for the second post')
#             )

# connection.commit()
# connection.close()

# noooooo, load df
ann_df = pd.read_csv('bag_of_words_translated - temp for figs.csv')
