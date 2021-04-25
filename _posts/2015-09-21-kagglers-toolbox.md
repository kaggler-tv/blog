# Kaggler’s Toolbox – Setup

![](/images/20150921-kagglers-toolbox.png)

I’d like to open up my toolbox that I’ve built for data mining competitions, and share with you.

Let me start with my setup.

1. TOC
{:toc}

# System

I have access to 2 machines:

* **Laptop** – Macbook Pro Retina 15″, OS X Yosemite, i7 2.3GHz 4 Core CPU, 16GB RAM, GeForce GT 750M 2GB, 500GB SSD
* **Desktop** – Ubuntu 14.04, i7 5820K 3.3GHz 6 Core CPU, 64GB RAM, GeForce GT 620 1GB, 120GB SSD + 3TB HDD

I purchased the desktop from eBay around at \$2,000 a year ago (September 2014).

# Git

As the code repository and version control system, I use git.

It’s useful for collaboration with other team members.  It makes easy to share the code base, keep track of changes and resolve conflicts when two people change the same code.

It’s useful even when I work by myself too.  It helps me reuse and improve the code from previous competitions I participated in before.

For competitions, I use gitlab instead of github because it offers unlimited number of private repositories.

# S3 / Dropbox

I use S3 to share files between my machines.  It is cheap – it costs me about \$0.1 per month on average.

To access S3, I use AWS CLI.  I also used to use s3cmd and like it.

I use Dropbox to share files between team members.

# Makefile

For flow control or pipelining, I use `makefiles` (or GNU `make`).

It modularizes the long process of a data mining competition into feature extraction, single model training, and ensemble model training, and controls workflow between components.

For example, I have a top level makefile that defines the raw data file locations, folder hierarchies, and target variable.

`Makefile`
```
# directories
DIR_DATA := data
DIR_BUILD := build
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST)

# data files for training and predict
DATA_TRN := $(DIR_DATA)/train.csv
DATA_TST := $(DIR_DATA)/test.csv
SAMPLE_SUBMISSION := $(DIR_DATA)/sample_submission.csv

ID_TST := $(DIR_DATA)/id.tst.csv
HEADER := $(DIR_DATA)/header.csv
CV_ID := $(DIR_DATA)/cv_id.txt

Y_TRN:= $(DIR_FEATURE)/y.trn.txt
Y_TST:= $(DIR_FEATURE)/y.tst.txt

$(DIRS):
	mkdir -p $@

$(HEADER): $(SAMPLE_SUBMISSION)
	head -1 $< > $@

$(ID_TST): $(SAMPLE_SUBMISSION)
	cut -d, -f1 $< | tail -n +2 > $@

$(Y_TST): $(SAMPLE_SUBMISSION) | $(DIR_FEATURE)
	cut -d, -f2 $< | tail -n +2 > $@

$(Y_TRN) $(CV_ID): $(DATA_TRN) | $(DIR_FEATURE)
	python src/extract_target_cvid.py --train-file $< \
                                      --target-file $(Y_TRN) \
                                      --cvid-file $(CV_ID)

# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
```

Then, I have makefiles for features that includes the top level makefile, and defines how to generate training and test feature files in various formats (CSV, libSVM, VW, libFFM, etc.).

`Makefile.feature.j3`
```
#--------------------------------------------------------------------------
# j3: h2 + row id
#--------------------------------------------------------------------------
include Makefile

FEATURE_NAME := j3

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.sps
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.sps

$(FEATURE_TRN) $(FEATURE_TST): $(DATA_TRN) $(DATA_TST) | $(DIR_FEATURE)
	python ./src/generate_$(FEATURE_NAME).py --train-file $< \
                                             --test-file $(word 2, $^) \
                                             --train-feature-file $(FEATURE_TRN) \
                                             --test-feature-file $(FEATURE_TST)

```

Then, I have makefiles for single model training that includes a feature makefile, and defines how to train a single model and produce CV and test predictions.

`Makefile.xg`
```
include Makefile.feature.j3

N = 10000
DEPTH = 6
LRATE = 0.05
SUBCOL = 1
SUBROW = 0.8
SUBLEV = 0.5
WEIGHT = 1
N_STOP = 100
ALGO_NAME := xg_$(N)_$(DEPTH)_$(LRATE)_$(SUBCOL)_$(SUBROW)_$(SUBLEV)_$(WEIGHT)_$(N_STOP)
MODEL_NAME := $(ALGO_NAME)_$(FEATURE_NAME)

METRIC_VAL := $(DIR_METRIC)/$(MODEL_NAME).val.txt

PREDICT_VAL := $(DIR_VAL)/$(MODEL_NAME).val.yht
PREDICT_TST := $(DIR_TST)/$(MODEL_NAME).tst.yht

SUBMISSION_TST := $(DIR_TST)/$(MODEL_NAME).sub.csv
SUBMISSION_TST_GZ := $(DIR_TST)/$(MODEL_NAME).sub.csv.gz

all: validation submission
validation: $(METRIC_VAL)
submission: $(SUBMISSION_TST)
retrain: clean_$(ALGO_NAME) submission

$(PREDICT_TST) $(PREDICT_VAL): $(FEATURE_TRN) $(FEATURE_TST) $(CV_ID) \
                                   | $(DIR_VAL) $(DIR_TST)
	./src/train_predict_xg.py --train-file $< \
                              --test-file $(word 2, $^) \
                              --predict-valid-file $(PREDICT_VAL) \
                              --predict-test-file $(PREDICT_TST) \
                              --depth $(DEPTH) \
                              --lrate $(LRATE) \
                              --n-est $(N) \
                              --subcol $(SUBCOL) \
                              --subrow $(SUBROW) \
                              --sublev $(SUBLEV) \
                              --weight $(WEIGHT) \
                              --early-stop $(N_STOP) \
                              --cv-id $(lastword $^)

$(SUBMISSION_TST_GZ): $(SUBMISSION_TST)
	gzip $<

$(SUBMISSION_TST): $(PREDICT_TST) $(HEADER) $(ID_TST) | $(DIR_TST)
	paste -d, $(lastword $^) $< > $@.tmp
	cat $(word 2, $^) $@.tmp > $@
	rm $@.tmp

$(METRIC_VAL): $(PREDICT_VAL) $(Y_TRN) | $(DIR_METRIC)
	python ./src/evaluate.py --predict-file $< \
                             --target-file $(word 2, $^) > $@
	cat $@


clean:: clean_$(ALGO_NAME)

clean_$(ALGO_NAME):
	-rm $(METRIC_VAL) $(PREDICT_VAL) $(PREDICT_TST) $(SUBMISSION_TST)
	find . -name '*.pyc' -delete

.DEFAULT_GOAL := all
```

Then, I have makefiles for ensemble features that defines which single model predictions to be included for ensemble training.

`Makefile.feature.esb3`
```
include Makefile

FEATURE_NAME := esb4

BASE_MODELS := xg_10000_6_0.05_1_0.8_0.5_1_100_h2 \
               xg_10000_6_0.05_1_0.8_0.5_1_100_j3 \
               keras_100_2_128_0.5_512_5_h2 \
               keras_100_2_128_0.5_512_5_j3

PREDICTS_TRN := $(foreach m, $(BASE_MODELS), $(DIR_VAL)/$(m).val.yht)
PREDICTS_TST := $(foreach m, $(BASE_MODELS), $(DIR_TST)/$(m).tst.yht)

FEATURE_TRN := $(DIR_FEATURE)/$(FEATURE_NAME).trn.csv
FEATURE_TST := $(DIR_FEATURE)/$(FEATURE_NAME).tst.csv

%.sps: %.csv
	python src/csv_to_sps.py --csv-file $< --sps-file $@

$(FEATURE_TRN): $(Y_TRN) $(PREDICTS_TRN) | $(DIR_FEATURE)
	paste -d, $^ | tr -d '\r' > $@

$(FEATURE_TST): $(Y_TST) $(PREDICTS_TST) | $(DIR_FEATURE)
	paste -d, $^ | tr -d '\r' > $@

clean:: clean_$(FEATURE_NAME)

clean_$(FEATURE_NAME):
	-rm $(FEATURE_TRN) $(FEATURE_TST)
```

Finally, I can (re)produce the submission from XGBoost ensemble with 9 single models described in `Makefile.feature.esb4` by (1) replacing include `Makefile.feature.j3` in `Makefile.xg` with include `Makefile.feature.esb4` and (2) running:

```bash
$ make -f Makefile.xg
```

# SSH Tunneling

When I’m connected to Internet, I always ssh to the desktop for its computational resources (mainly for RAM).

I followed Julian Simioni’s tutorial to allow remote SSH connection to the desktop.  It needs an additional system with a publicly accessible IP address.  You can setup an AWS micro (or free tier) EC2 instance for it.

# tmux

tmux allows you to keep your SSH sessions even when you get disconnected.  It also let you split/add terminal screens in various ways and switch easily between those.

Documentation might look overwhelming, but all you need are:

```bash
# If there is no tmux session:
$ tmux
```

or

```bash
# If you created a tmux session, and want to connect to it:
$ tmux attach
```

Then to create a new pane/window and navigate in between:

* Ctrl + b + " – to split the current window horizontally.
* Ctrl + b + % – to split the current window vertically.
* Ctrl + b + o – to move to next pane in the current window.
* Ctrl + b + c – to create a new window.
* Ctrl + b + n – to move to next window.

To close a pane/window, just type exit in the pane/window.

Hope this helps.

Next up is about machine learning tools I use.

Please share your setups and thoughts too. 🙂
