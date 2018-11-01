import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import Pipeline

class Plot:
    
    def __init__(self):
        self.df = Pipeline().df
        
    def most_common_words_artists(self):
        g = self.df.groupby('ARTIST')
        groups = list(g.groups.keys())
        ct = {i:pd.Series(sum([item for item in g.get_group(i).TOKENS], [])).value_counts() for i in groups}
        for key in ct.keys():
            artist = ct[key]
            plt.rcParams.update({'font.size': 22})
            plt.figure(figsize=(10,5))
            plt.barh(list(artist.index)[:11], artist.values[:11])
            plt.title('10 Most Common Words - {}'.format(key))
        plt.show()
        
    def most_common_words_all(self):
        ct = pd.Series(sum([item for item in self.df.TOKENS], [])).value_counts()
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(10,5))
        plt.barh(list(ct.index)[:11], ct.values[:11])
        plt.title('10 Most Common Words Overall')
        plt.show()
        
    def most_common_words_class(self):
        g = self.df.groupby('CLASS')
        groups = list(g.groups.keys())
        ct = {i:pd.Series(sum([item for item in g.get_group(i).TOKENS], [])).value_counts() for i in groups}
        for key in ct.keys():
            name = ''
            if key == 1:
                name = 'Spam'
            else:
                name = 'Normal(Not Spam)'
            artist = ct[key]
            plt.rcParams.update({'font.size': 22})
            plt.figure(figsize=(10,5))
            plt.barh(list(artist.index)[:11], artist.values[:11])
            plt.title('10 Most Common Words - {}'.format(name))
        plt.show()
        
