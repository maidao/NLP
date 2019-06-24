from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import numpy as np

def word_2_vec(train_data):
    model_w2v = Word2Vec([train_data], size=100, window=10, min_count=1, workers=4, sg=1)
    w1 = ['food', 'vietnamese']
    pred_mot = model_w2v.wv.most_similar(positive=w1)
    # pred_mot = model_w2v.wv.similar_by_word('food')
    print('Predict words with Word2Vec:')
    for i in pred_mot:
        print(i)

    print("----------------------------------------")
    # PCA và visualization: PCA giảm vector word từ 100 chiều về 2 chiều, để vẽ lên không gian 2 chiều
    words_np = []
    words_label = []
    for word in model_w2v.wv.vocab.keys():
        words_np.append(model_w2v.wv[word])
        words_label.append(word)

    pca = PCA(n_components=2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)

    # matplotlib inline
    plt.rcParams["figure.figsize"] = (20, 20)
    for index, vec in enumerate(reduced):
        if index < 200:
            x, y = vec[0], vec[1]
            plt.scatter(x, y)
            plt.annotate(words_label[index], xy=(x, y))
    plt.colorbar()
    # plt.show()

def fast_text(train_data):
    model_FastText = FastText([train_data], size=100, window=5, min_count=5, workers=4, sg=1)
    pred_mot_fast = model_FastText.wv.most_similar('food')
    print('Predict words with FastText:')
    for k in pred_mot_fast:
        print(k)

    print("----------------------------------------")

def bag_of_words(tweet_sorted):
    wordcloud = WordCloud().generate_from_frequencies(dict(tweet_sorted))
    plt.figure(figsize=(30, 30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.savefig('bag1.png', bbox_inches='tight')
    # plt.show()

def t_SNE(data_resto):
    X = data_resto[['replies_count','likes_count']]
    X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X)
    X_pca = PCA().fit_transform(X)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=data_resto.new_date)
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_resto.new_date)
    plt.colorbar()
    plt.show()

def k_means(data_resto):
    X = data_resto[['username','replies_count', 'likes_count']]
    Y = data_resto['new_date']
    estimators = [('k_means_8', KMeans(n_clusters=8)),
                  ('k_means_3', KMeans(n_clusters=3))]

    fignum = 1
    titles = ['8 clusters', '3 clusters']

    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 1], X[:, 0], X[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1
    plt.show()






