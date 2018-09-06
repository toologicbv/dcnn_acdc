import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import OrderedDict


def visualize_features(feature_projection, y_labels, extra_labels=None, exper_hdl=None, layer=None, do_save=False,
                       width=10, height=8, patient_id=None, label_color_type='base_apex'):

    if patient_id is not None:
        label_color_type = "patient_id"
    else:
        if label_color_type == "patient_id":
            raise ValueError("ERROR - Patient_id is not None but label_color_type != patient_id.")
    colors = ['b', 'g']
    mycmap = mpl.cm.get_cmap('Spectral')
    markers = ['x', 'D']
    legend_labels = ['normal', 'low-dice']
    plot_data = np.append(feature_projection, np.expand_dims(y_labels, axis=1), axis=1)
    data_points = np.empty((0, extra_labels.shape[1] + 1))
    fig = plt.figure(figsize=(width, height))
    # ax = plt.subplot(1, 1, 1)
    i = 0
    for x, y, c in plot_data.tolist():
        p_id = extra_labels[i, 0]
        if label_color_type == "patient_id":
            if p_id == patient_id:
                colour = 'r'
            else:
                colour = colors[int(c)]
        elif label_color_type == "base_apex":
                if extra_labels[i, 4] == 1:
                    colour = 'r'
                else:
                    colour = colors[int(c)]
        elif label_color_type == "base_apex_continuum":
            colour = mycmap(extra_labels[i, 5])

        plt.scatter(x, y, c=colour, marker=markers[int(c)], label=legend_labels[int(c)],
                    cmap=mycmap)
        if extra_labels is not None:
            if y_labels[i] == 1 or y_labels[i] == 0:
                dta_labels = np.zeros(extra_labels.shape[1] + 1).astype(np.int16)
                dta_labels[0] = int(i)
                dta_labels[1:] = extra_labels[i, :]
                data_points = np.vstack((data_points, dta_labels)) if data_points.size else dta_labels
                # plt.annotate(str(int(i)), (x, y))
        i += 1
    # remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.axis('off')
    plt.title("t-sne plot of features extracted from layer " + layer)
    plt.legend(by_label.values(), by_label.keys(), loc='lower left', numpoints=1, ncol=5, fontsize=8)

    if do_save:
        outfile = exper_hdl.exper.logdir + layer + '-tsne_projection.png'
        print("Save to %s" % outfile)
        plt.savefig(outfile)


    return data_points
