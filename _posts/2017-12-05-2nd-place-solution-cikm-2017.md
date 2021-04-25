# Second Place Solution at CIKM AnalytiCup 2017 – Lazada Product Title Quality Challenge

by Tam Nguyen

# Lazada Product Title Quality Challenge

In this challenge, the participants were provided with a set of product titles, description, and attributes, together with the associated title quality scores (clarity and conciseness) as labeled by Lazada internal QC team. The task is to build a product title quality model that can automatically grade the clarity and the conciseness of a product title.

# Team members

* Tam T. Nguyen, Kaggle Grandmaster, Postdoctoral Research Fellow at Ryerson University
* Hossein Fani, PhD Student at University of New Brunswick
* Ebrahim Bagheri, Associate Professor at Ryerson University
* Gilberto Titericz, Kaggle Grandmaster ( #1 Kaggler), Data Scientist at AirBnb

# Solution Overview

We present our winning approach for the Lazada Product Title Quality Challenge for the CIKM Cup 2017 where the data set was annotated as conciseness and clarity by Lazada QC team.

The participants were asked to build machine learning model to predict conciseness and clarity of an SKU based on product title, short description, product categories, price, country, and product type. As sellers could freely enter anything for title and description, they might contain typos or misspelling words. Moreover, there were many annotators labelling the data so there must be disagreement on the true label of an SKU. This makes the problem difficult to solve if one is solely using traditional natural language processing and machine learning techniques.

In our proposed approach, we adapted text mining and machine learning methods which take into account both feature and label noises. Specifically, we are using bagging methods to deal with label noise where the whole training data cannot be used to build our models. Moreover, we think that for each SKU, conciseness and clarity would be annotated by the same QC. It means that conciseness and clarity should be correlated in a certain manner. Therefore, we extended our bagging approach by considering out of fold leakage to take advantage of co-relation information.

Our proposed approach achieved the root mean squared error (RMSE) of 0.3294 and 0.2417 on the test data for conciseness and clarity, respectively. You may refer to the [paper](http://cikm2017.org/download/analytiCup/session3/CIKMAnalytiCup2017_LazadaProductTitleQuality_T2.pdf) or [source code](https://github.com/nthanhtam/cikmcup2017) for more details.

