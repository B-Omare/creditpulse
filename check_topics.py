import pandas as pd
df = pd.read_csv('reports/bertopic_summary.csv')
print(df[['Topic', 'Count', 'Name']].to_string())