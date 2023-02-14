import pandas as pd
import numpy as np

import statistics

statistics.mode([3,2,1])


s1 = pd.read_csv('./stuff/voting/sub1.csv')
# s11 = pd.read_csv('./stuff/voting/sub11.csv')
s2 = pd.read_csv('./stuff/voting/sub2.csv')
s3 = pd.read_csv('./stuff/voting/sub3.csv')

df = pd.DataFrame({
    's1': s1['label'],
    's2': s2['label'],
    's3': s3['label'],
})

s_voted = df.apply(statistics.mode, axis=1)

sum( (s1['label'] - s2['label']) == 0) / 100
sum( (s11['label'] - s_voted) == 0) / 100

s1['label'] = s_voted
s1.to_csv('./stuff/voting/submission_voted.csv', index=False)
