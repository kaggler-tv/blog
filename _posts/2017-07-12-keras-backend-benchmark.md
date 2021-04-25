# Keras Backend Benchmark: Theano vs TensorFlow vs CNTK

by Jeong-Yoon Lee

Inspired by [Max Woolf’s benchmark](http://minimaxir.com/2017/06/keras-cntk/), the performance of 3 different backends (Theano, TensorFlow, and CNTK) of Keras with 4 different GPUs (K80, M60, Titan X, and 1080 Ti) across various neural network tasks are compared.

For the performance of TensorFlow and CNTK with K80, the numbers reported at [Max Woolf’s benchmark](http://minimaxir.com/2017/06/keras-cntk/) are used.

# Conclusion

The accuracies of Theano, TensorFlow and CNTK backends are similar across all benchmark tests, while speeds vary a lot.

* **Theano** is significantly (up to 50 times) **slower** than TensorFlow and CNTK.
* Between TensorFlow and CNTK, **CNTK** is a lot (about 2 to 4 times) **faster** than TensorFlow for **LSTM** (Bidirectional LSTM on IMDb Data and Text Generation via LSTM), while speeds for other type of neural networks are close to each other.

Among K80, M60, Titan X and 1080 Ti GPUs:

* **1080 Ti** is the **fastest**.
* **K80** is the **slowest**.
* **M60** is **faster** than K80 and **comparable** to Titan X and 1080 Ti.
* **Theano** is significantly (up to 14 times) **faster** on **1080 Ti** than on Titan X, while the improvements for TensorFlow and CNTK are moderate.

Detailed results are available at https://github.com/szilard/benchm-dl/blob/master/keras_backend.md