# Data Science Career for Neuroscientists + Tips for Kaggle Competitions

![](/images/20131007-brain-circuit.jpg)

Recently Prof. Konrad Koerding at Northwestern University asked for an advice on his Facebook for one of his Ph.D student, who studies Computational Neuroscience but wants to pursue his career in Data Science.  It reminded me of the time I was looking for such opportunities, and shared my thoughts (now posted on the webpage of his lab [here](http://kordinglab.com/2016/01/05/leave-neuroscience.html)). I decide to post it here with a few fixes so that it can help others.

1. TOC
{:toc}

--—

# Introduction

First, I’d like to say that Data Science is a relatively new field (like Computational Neuroscience), and you don’t need to feel bad to make the transition after your Ph.D.  When I was out to the job market, I didn’t have any analytic background at all either.

I started my industrial career at one of analytic consulting companies, Opera Solutions in San Diego, where one of Nicolas‘ friends, Jacob, runs the R&D team of the company.  Jacob did his Ph.D under the supervision of Prof. Michael Arbib at University of Southern California in Computational Neuroscience as well.  During the interview, I was tested to prove my thought process, basic knowledges in statistics and Machine Learning, and programming, which I’d practiced through out my Ph.D everyday.

So, if he has a good Machine Learning background with programming skills (I’m sure that he does, based on the fact he’s your student), he can be competent to pursue his career in Data Science.

# Tools in Data Science

Back in the graduate school, I used mostly MATLAB with some SPSS and C.  In the Data Science field, Python and R are most popular languages, and SQL is a kind of necessary evil.

R is similar to MATLAB except that it’s free.  It is not a hardcore programming language and doesn’t take much time to learn.  It comes with the latest statistical libraries and provides powerful plotting functions.  There are many IDEs, which make easy to use R, but my favorite is R Studio.  If you run R on the server with R Studio Server, you can access it from anywhere via your web browser, which is really cool.  Although native R plotting functions are excellent by themselves, the ggplot2 library provides more eye-catching visualization.

For Python, Numpy + Scipy packages provides similar vector-matrix computation functionalities as MATLAB.  For Machine Learning algorithms, you need Scikit-Learn, and for data handling, Pandas will make your life easy.  For debugging and prototyping, iPython Notebook is really handy and useful.

SQL is an old technology but still widely used.  Most of data are stored in the data warehouse, which can be accessed only via SQL or SQL equivalents (Oracle, Teradata, Netezza, etc.).  Postgres and MySQL are powerful yet free, so it’s perfect to practice with.

# Hints for Kaggle Data Mining Competitions

Fortunately, I had a chance to work with many of top competitors such as the 1st and 2nd place teams at Netflix competitions, and learn how they do at competitions.  Here are some tips I found helpful.

## Don’t jump into algorithms too fast.

Spend enough time to understand data.  Algorithms are important, but no matter how good algorithm you use, garbage-in only leads to garbage-out.  Many classification/regression algorithms assume the Gaussian distributed variables, and fail to make good predictions if you provide non-Gaussian distributed variables.  So, standardization, normalization, non-linear transformation, discretization, binning are very important.

## Try different algorithms and blend.

There is no universal optimal algorithm.  Most of times (if not all), the winning algorithms are ensembles of many individual models with tens of different algorithms.  Combining different kinds of models can improve prediction performance a lot.  For individual models, I found Random Forest, Gradient Boosting Machine, Factorization Machine, Neural Network, Support Vector Machine, logistic/linear regression, Naive Bayes, and collaborative filtering are mostly useful.  Gradient Boosting Machine and Factorization Machine are often the best individual models.

## Optimize at last.

Each competition has a different evaluation metric, and optimizing algorithms to do the best for that metric can improve your chance to win.  Two most popular metrics are RMSE and AUC (area under the ROC curve).  Algorithms optimizing one metric is not the optimal for the other. Many open source algorithm implementations provide only RMSE optimization, so for AUC (or other metric) optimization, you need to implement it by yourself.
