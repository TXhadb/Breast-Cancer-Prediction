import pandas as pd

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

df = pd.read_csv('breast-cancer-wisconsin.data', names = column_names)
drop_rows = [i for i in range(len(df)) for j in column_names if df.at[i, j] == '?']
df = df.drop(drop_rows)

new_columns = [column_names[0]] + [column_names[-1]] + column_names[1:-1]
df = df.reindex(columns = new_columns)

df = df[df.columns[1:]]
df[df.columns[0]] -= 3

df.to_csv('data_file.csv', index = False)
