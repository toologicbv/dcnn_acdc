import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.manifold import TSNE


def visualize_features(features, labels, logpath=None, prefix=None):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', '#ff6666', '#66ff66', '#6666ff', '#666666']
    markers = ['x', '+', '^', 'D', '<', '>', '8', '*', 'H', 'o']
    legend_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    tsne = TSNE(n_components=2, random_state=0, init="pca")
    feature_projection = tsne.fit_transform(features)
    plot_data = np.append(feature_projection, np.expand_dims(labels, axis=1), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for x, y, c in plot_data.tolist():
        ax.scatter(x[:1000], y[:1000], c=colors[int(c)], marker=markers[int(c)], label=legend_labels[int(c)])

    # remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.axis('off')
    plt.title("t-sne plot of features extracted from layer " + prefix)
    plt.legend(by_label.values(), by_label.keys(), loc='lower left', numpoints=1, ncol=5, fontsize=8)
    outfile = logpath + prefix + '-tsne_projection.png'
    print("Save to %s" % outfile)
    plt.savefig(outfile)
