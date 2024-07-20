import pandas as pd
df = pd.read_csv('./content/advertising.csv')

data = df.to_numpy()

A = data[:, 3].mean()
nearest_sales_value = data[:, 3][(data[:, 3] - A).argmin()]
print(nearest_sales_value + A)

scores = []
for value in data[:, 3]:
    if value > nearest_sales_value + A:
        scores.append('Good')
    elif value < nearest_sales_value + A:
        scores.append('Bad')
    else:
        scores.append('Average')

print(scores[7:10])
