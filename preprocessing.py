"""
Cite 
https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
"""

import re
import glob
import gensim
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob, Word

class Pipeline():
    """
    Pipeline loads all data into multi-hierarchical df for later use.
    Features are extracted and preprocessing is performed.
    """
    
    def __init__(self):
        self.df = self.get_df()
        self.run_all()
        self.doc2vec_model = self.get_doc_vectors()
    
    def get_df(self):
        """
        Loads all csv files as df and adds ARTIST column
        Returns Concatenated df
        """
        names = ["Psy",
                 "KatyPerry",
                 "LMFAO",
                 "Eminem",
                 "Shakira"]
        paths = sorted(glob.glob("datasets/*"))
        dfs = []
        for i in range(5):
            df = pd.read_csv(paths[i])
            df['ARTIST'] = names[i]
            dfs.append(df)
            
        return pd.concat(dfs)
    
    def get_num_words(self):
        """
        Makes WORD_COUNT column in self.df
        """
        self.df['WORD_COUNT'] = self.df['CONTENT'].apply(lambda x: len(str(x).split(" ")))
    
    def get_num_chars(self):
        """
        Makes CHAR_COUNT column in self.df
        """
        self.df['CHAR_COUNT'] = self.df['CONTENT'].apply(lambda x: len(str(x)))
       
    def get_avg_word_len(self):
        """
        Makes AVG_WORD_LEN column in self.df
        """
        self.df['AVG_WORD_LEN'] = self.df['CONTENT'].apply(lambda x: sum(len(i) for i in str(x).split(" "))/len(str(x).split(" ")))
        
    def get_num_stopwords(self):
        """
        Makes NUM_STOPWORDS column in self.df
        """
        stop = stopwords.words('english')
        self.df['NUM_STOPWORDS'] = self.df['CONTENT'].apply(lambda x: len([x for x in x.split() if x in stop]))
        
    def get_has_link(self):
        """
        Binary encodes and Makes HAS_LINK columns in self.df
        IF HAS_LINK:
            1
        ELSE:
            0
        credit for regex
        - https://gist.github.com/gruber/249502
        - https://daringfireball.net/2010/07/improved_regex_for_matching_urls
        """
        WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
        self.df['HAS_LINK'] = self.df['CONTENT'].apply(lambda x: 1 if len(re.findall(WEB_URL_REGEX, x)) > 0 else 0 )
        
    def get_has_channel(self):
        """
        Binary encodes and Makes HAS_CHANNEL columns in self.df
        If HAS_CHANNEL(word):
            1
        ELSE:
            0
        """
        self.df['HAS_CHANNEL'] = self.df['TOKENS'].apply(lambda x: 1 if 'channel' in x else 0)
    def get_has_subscribe(self):
        """
        Binary encodes and Makes HAS_CHANNEL columns in self.df
        If HAS_SUBSCRIBE(word):
            1
        ELSE:
            0
        """
        self.df['HAS_SUBSCRIBE'] = self.df['TOKENS'].apply(lambda x: 1 if 'subscribe' in x else 0)
        
    
    def get_sentiment(self):
        """
        Makes POLARITY and SUBJECTIVITY columns
        """
        sentiments = list(self.df['CONTENT'].apply(lambda x: TextBlob(x).sentiment))
        self.df['SUBJECTIVITY'] , self.df['POLARITY'] = zip(*sentiments)
        
    def get_term_frequency_table(self):
        pass

    def get_doc_vectors(self):
        corpus = []
        for i, line in enumerate(self.df.CONTENT.values):
            corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]))
        
        
        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print("Getting DOC2VECTORS")
        self.df['DOC2VEC'] = [model[i] for i in range(len(corpus))]
        return model
        

    def run_all(self):
        """
        run_all() is catch all function that 
        preprocesses self.df['CONTENT']
        runs feature extraction that requires clean data
        INCLUDES in order
        self.get_num_words()
        self.get_num_chars()
        self.get_avg_word_len()
        self.get_num_stopwords()
        self.get_has_link()
        lowercase all text
        removes punctuation
        removes stop words
        spelling correction
        Makes TOKENS column by tokenization
        lemmatization
        self.get_has_channel()
        self.get_has_subscribe()
        THEN
        removes most frequent and rare words
        """
        print("Makes WORD_COUNT column")
        self.get_num_words()
        print("Makes CHAR_COUNT column")
        self.get_num_chars()
        print("Makes AVG_WORD_LEN column")
        self.get_avg_word_len()
        print("Makes NUM_STOPWORDS column")
        self.get_num_stopwords()
        print("Makes HAS_LINK column")
        self.get_has_link()
        print("Lowercase All CONTENT")
        self.df['CONTENT'] = self.df['CONTENT'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        print("Removing Punctuation")
        self.df['CONTENT'] = self.df['CONTENT'].str.replace('[^\w\s]','')
        print("Removing Stopwords")
        stop = stopwords.words('english')
        self.df['CONTENT'] = self.df['CONTENT'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        #print("Spelling Corrections") 
        #TAKES TOO LONG
        #self.df['CONTENT'] = self.df['CONTENT'].apply(lambda x: str(TextBlob(x).correct()))
        print("Lemmatizing")
        self.df['CONTENT'] = self.df['CONTENT'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        print("Creating TOKENS Column")
        self.df['TOKENS'] = self.df['CONTENT'].apply(lambda x: list(TextBlob(x).words))
        print("Making HAS_CHANNEL column")
        self.get_has_channel()
        print("Making HAS_SUBSCRIBE column")
        self.get_has_subscribe()
        print("Making POLARITY and SUBJECTIVITY COLUMNS")
        self.get_sentiment()
        
df = Pipeline().df
print(df.DOC2VEC.head(10))

