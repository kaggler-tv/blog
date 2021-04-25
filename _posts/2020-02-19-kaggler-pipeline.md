# Kaggler Pipeline for Data Science Competitions

In this blog, we are going to go over the fundementals of the [Kaggler repository][Kaggler_repo], a machine learning pipeline for data science competitions. The Kaggler pipeline uses Makefiles and Python scripts to coordinate dependencies, and allows quick iteration of new features and models. You can watch the demo at [Kaggler TV Episode #4][KagglerTV_4].

The pipeline is driven by running a model training, such as logistic regression, using the corresponding Makefile, e.g.,\
`$make -f Makefile.logreg1`. Before going into the details of a model run, let's build our data and features from bottom up. To start, we need to initialize our repo by going to the [Kaggler repository][Kaggler_repo] and clicking **use this template**
 to name and create our repository, e.g., **cat-in-the-dat-ii**. Next, clone the repository by  
 
 ```
 $git clone https://github.com/YOUR_GITHUB_ID/cat-in-the-dat-ii.git
 $cd cat-in-the-dat-ii
 ```


## Data
The file [___`Makefile`___][makefile_github] defines the directories and the structure of the pipeline. 
```make
# XXX: competition name
COMPETITION := cat-in-the-dat-ii

# gsed on macOS. sed on LINUX
SED := gsed

# directories
DIR_DATA := input
DIR_BUILD := build
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst
DIR_SUB := $(DIR_BUILD)/sub

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_SUB)

# data files for training and predict
DATA_TRN := $(DIR_DATA)/train.csv
DATA_TST := $(DIR_DATA)/test.csv
SAMPLE_SUBMISSION := $(DIR_DATA)/sample_submission.csv

LABEL_IDX = 25

ID_TST := $(DIR_DATA)/id.tst.csv
HEADER := $(DIR_DATA)/header.csv

Y_TRN:= $(DIR_FEATURE)/y.trn.txt
Y_TST:= $(DIR_FEATURE)/y.tst.txt

data: $(DATA_TRN) $(DATA_TST) $(SAMPLE_SUBMISSION)

$(DIRS):
        mkdir -p $@

$(DATA_TRN) $(DATA_TST) $(SAMPLE_SUBMISSION): | $(DIR_DATA)
        kaggle competitions download -c $(COMPETITION) -p $(DIR_DATA)
        find . -name "*.zip" -exec sh -c 'unzip -d `dirname {}` {}' ';'

$(HEADER): $(SAMPLE_SUBMISSION)
        head -1 $< > $@

$(ID_TST): $(SAMPLE_SUBMISSION)
        cut -d, -f1 $< | tail -n +2 > $@

$(Y_TST): $(SAMPLE_SUBMISSION) | $(DIR_FEATURE)
        cut -d, -f2 $< | tail -n +2 > $@

$(Y_TRN): $(DATA_TRN) | $(DIR_FEATURE)
        cut -d, -f$(LABEL_IDX) $< | tail -n +2 > $@

# cleanup
clean::
        find . -name '*.pyc' -delete

clobber: clean
        -rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup

```
First, we need to define the name of the competition in ___`Makefile`___. After that, running `$make data` will download the specified competition data from Kaggle into the **./input** directory. You need to install the [Kaggle API][Kaggle_API], and accept the competition rules on Kaggle to be able to download the data. If you do not download the data manually at this time, the pipeline will automatically start the download when running the first model training. The parameters defined in ___`Makefile`___ are as follows.

  
   `$DIR_DATA` is the directory for the input data. 
  
   `$DIR_TRN`, `$DIR_TST`, and `$SAMPLE_SUBMISSION` are the downloaded train, test and sample submission files. 
  
   `LABEL_IDX` is the column index of the *target* variable in the train file, and needs to be specified. 

   `$Y_TRN` is the file containing the target labels, and it is created automatically by the pipeline.
  
   `$HEADER` and `$ID_TST` are also created by the pipeline, and are used to build submission files.

## Feature Engineering
Let's create our first feature by one hot encoding all the categorical columns [^5]. All the columns in this competition are categorical. The feature engieering for a specific feature are defined in files **./src/generate_$FEATURE_NAME.py**, e.g., [**./src/generate_e1.py**][e1_github]. We also need to create a makefile correponding to this feature, **Makefile.feature.e1** as follows.

```make
#--------------------------------------------------------------------------
# e1: all OHE'd features 
#--------------------------------------------------------------------------
include Makefile

FEATURE_NAME := e1

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps
FEATURE_MAP := $(DIR_FEATURE)/$(FEATURE_NAME).fmap

$(FEATURE_TRN) $(FEATURE_TST) $(FEATURE_MAP): $(DATA_TRN) $(DATA_TST) | $(DIR_FEATURE)
        python ./src/generate_$(FEATURE_NAME).py --train-file $< \
                                             --test-file $(lastword $^) \
                                             --train-feature-file $(FEATURE_TRN) \
                                             --test-feature-file $(FEATURE_TST) \
                                             --feature-map-file $(FEATURE_MAP)
```
Feature makefiles include all the parameters from ___`Makefile`___. The parameters defined in ___`Makefile.feature.e1`___ are

`FEATURE_NAME`: specified name of the feature  

`FEATURE_TRN`, `FEATURE_TST`: train and test feature files, which are the outputs created by **./src/generate_$(FEATURE_NAME).py**.

`FEATURE_MAP`: a file where we keep the name of the features, which is also created by **./src/generate_$(FEATURE_NAME).py**.

## Cross-Validation and Test Predictions
The models are defined in makefiles ___`Makefile.$ALGO_NAME`___, e.g., [___`Makefile.logreg1`___][makefile_logreg1_github]. At the top of each model file, we define which feature is going to be included as shown below. Then, we give the algorithm a short name, `ALGO_NAME`, for reference. We define the parameters for the algorithm, `C, REGULARIZER, CLASS_WEIGHT` and `SOLVER` in this case. We also specifiy a model name for reference, `MODEL_NAME`. The cross validation for algorithms are run by files **./src/train_predict_$MODEL_NAME.py**, e.g., **./src/train_predict_logreg1.py**, which produce validation and test predictions, `PREDICT_VAL` and `PREDICT_TST`. 

After the cross validation, **./src/evaluate.py** evaluates the validation predictions for a given metric, and writes the score to the file `METRIC_VAL`. Finally the submission file, `SUBMISSION_TST`, is created using the test predictions.

```make
include Makefile.feature.e1

ALGO_NAME := logreg
C := 1.0
REGULARIZER := l2
CLASS_WEIGHT := balanced
SOLVER := lbfgs
MODEL_NAME := $(FEATURE_NAME)_$(ALGO_NAME)_$(REGULARIZER)_$(C)

METRIC_VAL := $(DIR_METRIC)/$(MODEL_NAME).val.txt
PREDICT_VAL := $(DIR_VAL)/$(MODEL_NAME).val.yht
PREDICT_TST := $(DIR_TST)/$(MODEL_NAME).tst.yht
SUBMISSION_TST := $(DIR_SUB)/$(MODEL_NAME)_sub.csv

all: validation submission
validation: $(METRIC_VAL)
submission: $(SUBMISSION_TST)
retrain: clean_$(ALGO_NAME) submission

submit: $(SUBMISSION_TST)
        kaggle competitions submit -c $(COMPETITION) -f $< -m $(MODEL_NAME)

$(PREDICT_TST) $(PREDICT_VAL): $(FEATURE_TRN) $(FEATURE_TST) | $(DIR_VAL) $(DIR_TST)
        python ./src/train_predict_logreg1.py --train-feature-file $< \
                                         --test-feature-file $(word 2, $^) \
                                         --predict-valid-file $(PREDICT_VAL) \
                                         --predict-test-file $(PREDICT_TST) \
                                         --C $(C) \
                                         --regularizer $(REGULARIZER) \
                                         --class_weight $(CLASS_WEIGHT) \
                                         --solver $(SOLVER) \
                                         --retrain

$(METRIC_VAL): $(PREDICT_VAL) $(Y_TRN) | $(DIR_METRIC)
        python ./src/evaluate.py --predict-file $< \
                                 --target-file $(lastword $^) > $@
        cat $@

$(SUBMISSION_TST): $(PREDICT_TST) $(HEADER) $(ID_TST) | $(DIR_SUB)
        paste -d, $(lastword $^) $< > $@.tmp
        cat $(word 2, $^) $@.tmp > $@
        rm $@.tmp

.DEFAULT_GOAL := all

```

If we would like to run the same model with a different feature, e.g., [j1][j1_github], all we need to do is to change the first line in ___`Makefile.logreg1`___ to `include Makefile.feature.j1`. The pipeline will automatically create this feature and run cross validation using **./src/train_predict_j1.py**.

Similarly, if we would like to run cross-validation using a different model, such as LightGBM, we need to include the right feature in [___`Makefile.lgb1`___][makefile_lgb1_github], and run `$make -f Makefile.lgb1`. If the train and test features for are already created, they will not be created again.


## Ensemble
After creating several features and model runs, running the ensemble model is similar to running a single model. Ensemble model uses the predictions from single model runs as features. All we need to do is specify which model predictions should be included in the ensemble in [___`Makefile.feature.esb1`___][makefile_esb1_github] as base models. The feature names should be the same as the model names defined in the model makefiles. 

## Submit
Final step is to submit our predictions. Kaggler pipeline allows submitting predictions through CLI. You need to have the following lines in your model makefile, as in [___`Makefile.lgb1`___][makefile_lgb1_github]
```
submit: $(SUBMISSION_TST)
 kaggle competitions submit -c $(COMPETITION) -f $< -m $(MODEL_NAME)
```
To make a submission with the predictions from this model and feature, all you need to do is type the following. The submission will inculde `MODEL_NAME` as a message for the submission.

```
$make -f Makefile.lgb1 submit
```


## Conclusion
We covered the main components of the Kaggler repository. Hopefully, this blog helps you become more comfortable with the Kaggler pipeline. Happy Kaggling :)

## References

1. Kaggler repository: https://github.com/kaggler-tv/kaggler-template

2. Kaggler-TV Episode 4: https://www.youtube.com/watch?v=861NAO5-XJo&feature=youtu.be

3. Official Kaggle_API: https://github.com/Kaggle/kaggle-api

4. Kaggler template for cat-in-the-dat-ii: https://github.com/kaggler-tv/cat-in-the-dat-ii

5. https://www.kaggle.com/cuijamm/simple-onehot-logisticregression-score-0-80801



[Kaggler_repo]: https://github.com/kaggler-tv/kaggler-template

[KagglerTV_4]: https://www.youtube.com/watch?v=861NAO5-XJo&feature=youtu.be

[Kaggle_API]: https://github.com/Kaggle/kaggle-api

[^5]: https://www.kaggle.com/cuijamm/simple-onehot-logisticregression-score-0-80801

[makefile_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/Makefile

[e1_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/src/generate_e1.py

[j1_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/src/generate_j1.py

[makefile_e1_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/Makefile.feature.e1

[makefile_lgb1_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/Makefile.lgb1

[makefile_logreg1_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/Makefile.logreg1

[makefile_esb1_github]: https://github.com/kaggler-tv/cat-in-the-dat-ii/blob/master/Makefile.feature.esb1
