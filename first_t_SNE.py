from sklearn.manifold import TSNE
from main import *
import numpy as np
import seaborn as sn

data = dict(tweet_sorted)
X = []
Y = []

for i in data.values():
    X.append(i)
for j in data.keys():
    Y.append(j)

X_train = np.asarray(X)
Y_train = np.asarray(Y)

model = TSNE(n_components=2, random_state=0)
t_sne_data = model.fit_transform(X_train.reshape(-1, 1))
t_sne_data = np.vstack((t_sne_data.T,Y_train)).T

t_sne_df = pd.DataFrame(data=t_sne_data, columns=("Dim_1", "Dim_2", "label"))

sn.FacetGrid(t_sne_df, hue="label", height=6).map(plt.scatter,"Dim_1", "Dim_2")
plt.show()
