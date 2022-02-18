import pandas as pd
import json

df = pd.read_csv('./static/data/data.csv')
print(df.columns)

res = df.to_json(orient="records")
parsed = json.loads(res)
print(parsed)