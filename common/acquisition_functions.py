import numpy as np


def bald_function(probabilities):

    """
    Bayesian Active Learning by Disagreement (BALD)
        Mutual information between predictions and model posterior (Houlsby et al., 2011)
        Implemented as described in Y. Gal 2017 Deep Bayesian Active Learning with Image data

    :param probabilities: [mc samples, #classes, width, height] the Softmax probabilities for one
    image slice, for each of the classes

    :return: for each image pixel the BALD measure [width, height]
    """
    # Expected (w.r.t. p(w)) posterior Entropy of output (y) given image and D(training data)
    # \E_{p(\omega | \mathcal{D}_{train})} [\mathbb{H}[y | x, \omega] ]
    # inside: sum over classes (axis=1), outside: average over mc samples
    expected_entropy = - np.mean(np.sum(probabilities * np.log(probabilities + 1e-10), axis=1), axis=0)
    # first calculate mean softmax probabilities per pixel per class
    expected_p = np.mean(probabilities, axis=0)
    # compute Entropy of p(y | x, D(train)). Sum over the classes e.g. axis=0 (because we lost the original dim0
    # when computing the mean above)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=0)
    bald_acq = entropy_expected_p - expected_entropy
    # print('BALD_acq on first 10 points', BALD_acq[:10])

    return bald_acq

