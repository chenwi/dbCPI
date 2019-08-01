# dbCPI: A database of chemical and protein interactions based on literature mining

* Task description: http://www.biocreative.org/tasks/biocreative-vi/track-5/
* Data : http://www.biocreative.org/accounts/login/?next=/resources/corpora/chemprot-corpus-biocreative-vi/


## Requirements

* Python 3.6
* TensorFlow >=1.14.0
* NLTK


## Usage

### Update configuration file
 Go through the config file `config.py` to modify the
 parameters.

### Preprcessing
```
python preprocess/datahelper.py
```

By default, you will see the relation instances of train, dev and test
sets.

### Train and test

Load the sentences, initalize model parameters and run training and testing on
the dataset by:

```
train:
    python train.py
test:
    python run_demo.py
```

Our result is:

```
Precision: 0.724
Recall: 0.650
F-score: 0.685
```
