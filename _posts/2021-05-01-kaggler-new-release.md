---
title: "Kaggler v0.9.4 Release with Stacked DAE"
description: "Kaggler v0.9.4 has been released with the enhanced version of Denoising AutoEncoder (DAE)"
layout: post
toc: false
comments: true
hide: false
search_exclude: false
categories: [Kaggler, DAE]
---

Today, [`Kaggler`](https://github.com/jeongyoonlee/Kaggler) v0.9.4 is released with additional features for DAE as follows:
- In addition to the swap noise (`swap_prob`), the Gaussian noise (`noise_std`) and zero masking (`mask_prob`) have been added to DAE to overcome overfitting.
- Stacked DAE is available through the `n_layer` input argument (see Figure 3. in [Vincent et al. (2010), "Stacked Denoising Autoencoders"](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) for reference).

For example, to build a stacking DAE with 3 pairs of encoder/decoder and all three types of noises, you can do:
```python
from kaggler.preprocessing import DAE

dae = DAE(cat_cols=cat_cols, num_cols=num_cols, n_layer=3, noise_std=.05, swap_prob=.2, masking_prob=.1)
X = dae.fit_transform(pd.concat([trn, tst], axis=0))
```

If you're using previous versions, please upgrade `Kaggler` using 
```bash
pip install -U kaggler.
```

You can find Kaggle notebooks featured with [`Kaggler`](https://github.com/jeongyoonlee/Kaggler)'s DAE as follows:
- [Kaggler DAE + AutoLGB Baseline](https://www.kaggle.com/jeongyoonlee/kaggler-dae-autolgb-baseline): shows how to train a LightGBM model using `Kaggler`'s DAE features and `AutoLGB` model at the current TPS May competition
- [DAE with 2 Lines of Code with Kaggler](https://www.kaggle.com/jeongyoonlee/dae-with-2-lines-of-code-with-kaggler): shows how to use `Kaggler`'s DAE with the previous TPS April competition.

Any feedbacks, suggestions, questions for the package are welcome.

Hope it helps! :)
