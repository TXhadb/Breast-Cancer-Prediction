import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

print('samples: ' + str(len(df)))
print('malignant : benign = ' + str(round(len(df[df[df.columns[0]] == 1]) / len(df), 4))
      + ' : ' + str(round(len(df[df[df.columns[0]] == -1]) / len(df), 4)))
