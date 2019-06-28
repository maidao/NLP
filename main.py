from clean_data import *
from model_appied import *


# 1--- Load data
path = 'food.csv'
data_raw = Clean_data(path)
data_resto = data_raw.clean()
#data_resto.to_csv('resto.csv')
m = max(data_resto.likes_count)
print("max like: ",m)
print("max rep:", max(data_resto.replies_count))
like_most = data_resto[data_resto['likes_count'] == 14082]
rep_most = data_resto[data_resto['replies_count'] == 793]
print(rep_most[['user_id', 'username', 'link']])

print("----------------------------------------")

# 2--- Xóa những dấu thua
tweet_raw = data_resto['tweet'].to_string()
tweet_cleaned = data_raw.remove_ponctuation(tweet_raw)
tweed_new = data_raw.remove_mot_useless(tweet_cleaned)
tweet_sorted = data_raw.count_mot(tweed_new)

"""
f = open("tweet_sorted.txt","w+")
for i in tweet_sorted:
     f.write("%s\r\n" % str(i))
f.close()
"""

print('len of raw_tweet:',len(tweet_raw))
print('len of new_tweet:',len(tweed_new))
print('tweet_sorted:',tweet_sorted)

print("----------------------------------------")

# 4. NLP
train_data = tweed_new
#pred_w2v = word_2_vec(train_data)
#pred_fast_text = fast_text(train_data)
#pred_bag_of_word = bag_of_words(tweet_sorted)
#pred_tSNE = t_SNE(data_resto)
#pred_kMeans = k_means(data_resto)
pred_sentiment = blob(tweet_raw)
sen = data_raw.count_mot(pred_sentiment)
labels = 'Positive', 'Negative'
sizes = [1339/(1339+213), 213/(1339+213)]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
print(texts)
autotexts[0].set_fontsize(16)
autotexts[1].set_fontsize(16)
texts[0].set_fontsize(18)
texts[1].set_fontsize(18)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('sentiment.png')
plt.show()







