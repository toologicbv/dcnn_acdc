from sklearn.manifold import TSNE
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import cifar10_utils

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)


def visualize_features(features, labels, logpath=None, prefix=None):

    mpl.use('Agg')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', '#ff6666', '#66ff66', '#6666ff', '#666666']
    markers = ['x', '+', '^', 'D', '<', '>', '8', '*', 'H', 'o']
    legend_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    tsne = TSNE(n_components=2, random_state=0, learning_rate=1000)
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

prefix = "fc2"
file_prefix = "test_" + prefix
ROOT_PATH = "./server_checkpoints/convnet1/"
DEFAULT_INFILE= ROOT_PATH + file_prefix + "_output.npz"
npzfile = np.load(DEFAULT_INFILE)
feature_data = npzfile[prefix + "_output"]
if feature_data.ndim > 2:
    feature_data = np.squeeze(feature_data, axis=0)
# feature_data = npzfile[prefix]
cifar_x = cifar10.test.images
cifar_y = cifar10.test.labels
print(feature_data.shape)
y = np.argmax(cifar_y, axis=1)
print(y.shape)
labels = [float(x) for x in list(np.argmax(cifar_y, 1))]

visualize_features(feature_data, labels, logpath=ROOT_PATH, prefix=prefix)