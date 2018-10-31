import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import Pipeline


raw = Pipeline().df
g = raw.groupby('ARTIST')
groups = list(gp.groups.keys())
ct_dict = {i:pd.Series(sum([item for item in g.get_group(i).TOKENS], [])).value_counts() for i in groups}

print(ct_dict)