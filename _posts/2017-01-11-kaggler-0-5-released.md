# Kaggler 0.5.0 Released

by Jeong-Yoon Lee

I am glad to announce the release of [Kaggler](https://github.com/jeongyoonlee/Kaggler) 0.5.0. Kaggler 0.5.0 has a significant improvement in the performance of the FTRL algorithm thanks to Po-Hsien Chu ([github](https://github.com/stegben), [kaggle](https://www.kaggle.com/stegben), [linkedin](https://www.linkedin.com/in/benjamin-po-hsien-chu-32622687)).

# Results

We increase the train speed by up to 100 times compare to 0.4.x. Our benchmark shows that one epoch with 1MM records with 8 features takes 1.2 seconds with 0.5.0 compared to 98 seconds with 0.4.x on an i7 CPU.

# Motivation

The FTRL algorithm has been a popular algorithm since its first appearance on a [paper](http://www.jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf) published by Google. It is suitable for highly sparse data, so it has been widely used for click-through-rate (CTR) prediction in online advertisement. Many Kagglers use FTRL as one of their base algorithms in CTR prediction competitions. Therefore, we want to improve our FTRL implementation and benefit Kagglers who use our package.

# Methods

We profile the code with cProfile and resolve the overheads one by one:

* [Remove over-heads of Scipy Sparse Matrix row operation](https://github.com/jeongyoonlee/Kaggler/pull/21): Scipy sparse matrix checks many conditions in `__getitems__`, resulting in a lot of function calls. In `fit()`, we know that we’re fetching exactly each row, and it is very unlikely to exceed the bound, so we can fetch the indexes of each row in a [faster way](https://github.com/jeongyoonlee/Kaggler/pull/21/files#diff-68e5130bfd4c99a2d78351c2749555d9R136). This enhancement makes our FTRL 10x faster.
* [More c-style enhancement](https://github.com/jeongyoonlee/Kaggler/pull/22): Specify types more clearly, return a whole list instead of yielding feature indexes, etc. These enhancements make our FTRL 5X faster when `interaction==False`.
* [Faster hash function for interaction features](https://github.com/jeongyoonlee/Kaggler/pull/28): The last enhancement is to remove the overhead of hashing of interaction features. We use MurMurHash3, which scikit-learn uses, to directly hash the multiplication of feature indexes. This enhancement makes our FTRL 5x faster when `interaction==True`.

# Contributor

Po-Hsien Chu ([github](https://github.com/stegben), [kaggle](https://www.kaggle.com/stegben), [linkedin](https://www.linkedin.com/in/benjamin-po-hsien-chu-32622687))
