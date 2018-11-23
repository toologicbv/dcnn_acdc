# dcnn_acdc project

## (Bayesian) Dilated Convolutional Network

Deep learning segmentation model for cardiac MRI tissue segmentation (LV, RV myocardium cavity).
Base model is from Jelmer Wolterink.
Extended to Bayesian DCNN by means of MC dropout.

Trained and evaluated on ACDC and HVSMR datasets. 
Compared and evaluated Bayesian uncertainty maps with Entropy maps.
We also compared the effect of different loss functions on the calibration of the softmax probabilities.

### 23 November 2018

This is the last version which incorporates a region detector to filter segmentation errors based on uncertainty maps.

