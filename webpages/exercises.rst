
You can find the starting code at `the gitlab repository am4ip-lab3 <https://gitlab.com/am4ip/am4ip-lab3>`_

Exercise - Implementation of CBD Network
========================================
The goal of this lab is to implement Convolutional Blind Denoising (CBD) Network from `this arXiv paper <https://arxiv.org/pdf/1807.04686v2.pdf>`_
The am4ip library is composed of the following modules:

::

    am4ip
    │   datasets.py
    │   losses.py
    │   metrics.py
    │   models.py
    │   trainer.py
    │   utils.py
    │   utils.py

The role of `datasets.py` is to prepare the data. This includes data loading and pre-processing.

`losses.py` contains loss functions that are computed to train the proposed denoising network.

`metrics.py` which will IQ metrics from lab 1.

`models.py` will contain all implemented models, as well as a code skeleton for CBD Network.

`trainer.py` contains a utility function to perform the training.

Finally, `utils.py` contains a set of tools and utility function that are helpful, but not mandatory to used.

Activity: Extend the python script `train.py` in the script folder (or start with a notebook from scratch) as follows:

`train.py` is a skeleton of the code, including data loading into batches, the overall training procedure, etc.
Your goal is to implement different parts of this training procedure, including the model, the losses, the visualization, etc.

1. Load TID2013 dataset
2. Build CBD Network as presented in the course
3. Compute the loss as a reconstruction loss (MSE) and KL divergence
4. Perform the training
5. Compute IS and FID scores
6. Show some generated images

Additional Exercises
====================
1. Implement IQ metrics not implemented during lab 1 to evaluated the generated images
2. Modify existing code of TID2013 to be split between training and evaluation
3. Change the architecture of CBD Network and try to improve performances

