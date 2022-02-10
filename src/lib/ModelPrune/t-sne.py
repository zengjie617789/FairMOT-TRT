from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def get_tsne_figure(data_path, save_path='./model_outputs/figure'):
    tsne = TSNE(n_components=2)

    features_data = np.load(data_path, allow_pickle=True)
    print(features_data.shape)

    features_std = StandardScaler().fit_transform(features_data)
    fea_tsne = tsne.fit_transform(features_std)
    y = list(range(features_data.shape[0]))
    x_tsne_data = np.vstack((fea_tsne.T, y)).T
    df_tsne = pd.DataFrame(x_tsne_data)
    print(df_tsne)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, x=0, y=1, hue=2)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'total.jpg'))
    plt.show()

def main():
    data_path = './model_outputs/feature_data/total.npy'
    get_tsne_figure(data_path)

if __name__ == '__main__':
    main()