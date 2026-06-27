import pandas as pd
df = pd.read_csv('reports/fairness_metrics.csv')
print(df.to_string())