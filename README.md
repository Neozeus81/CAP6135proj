# CAP6135proj RNN network research

# Overview

The point of this project it to replicated and emulate the findings of this paper: [A Deep Learning Approach for Intrusion Detection Using Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8066291)
Which is how machine learning models can be used to detect and classify malicous network traffic for intrusion detection systems.

We created a model in each of the subdirectories to test their performance on the [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) dataset. 
The main model we spent the most time on is in RNN. We itereated over all of the hyper paramenters mentioned in the paper and also compiled the accuracies at each epoch using custom call back functions.


