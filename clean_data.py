import pandas as pd
import re
from nltk import ngrams
import nltk
import datetime

class Clean_data:
    def __init__(self, path):
        self.data_raw = pd.read_csv(path,low_memory=False)

    def clean(self):
        # select columns necessary
        data = self.data_raw[['date','user_id','username',
                                    'tweet','replies_count',
                                    'likes_count','link']]
        d = data['date']
        new_date = []
        for i in d:
            date_time_obj = datetime.datetime.strptime(i, '%Y-%m-%d')
            new_date.append(date_time_obj.year)
        data.loc[data.index,'new_date'] = new_date

        return data

    def remove_ponctuation(self,text):
        text = re.sub("[!@#$?.0-9^*%//()]","",text)
        #text = re.sub(r"[\.,\?,\@]+$", "", text)
        text = text.replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("“", " ") \
            .replace(":", " ").replace("”", " ") \
            .replace('"', " ").replace("'", " ") \
            .replace("!", " ").replace("?", " ")
        text.strip() #---> xóa khoảng trắng ở đầu và cuối ch
        return text


    def resto_ngram(self,string,n):
        # list(ngrams("Trăm năm trong cõi người ta".split(), 2))
        gram_str = list(ngrams(string, n))
        return gram_str

    def remove_mot_useless(self,text):
        remove_words = nltk.corpus.stopwords.words('english')
        new_tweed = []
        for i in text.split():
            if i.lower() not in remove_words and len(i) > 3 and i.isdigit() == False:
                new_tweed.append(i.lower())
        return new_tweed

    def count_mot(self,text):
        count_mot = {}
        for i in text:
            count_mot[i] = text.count(i)

        sorted_mot = sorted(count_mot.items(), key = lambda x: x[1], reverse = True)
        return sorted_mot













