# canon
Exploring the Pāli Canon with Machine Learning

## Installation

Use of [Anaconda](https://www.continuum.io/) is recommended.

- Create an Anaconda environment: `conda create --name canon python=3`
- Activate the environment: `conda activate canon`
- Install Anaconda packages: `conda install nltk tensorflow scikit-learn matplotlib`
- Install non-Anaconda packages: `pip install bs4`

## Text Processing

Run `python process.py` to download the [ATI archive](http://www.accesstoinsight.org/tech/download/bulk.html) (if needed) and process text into `data/sentences.py`

## Analysis

Run `python analyze.py` to train and evaluate a word vector model on the processed sentences.  This will produce some examples for evaluation and save a `tsne.png` file that contains the t-SNE plot for the results.  Subsequent runs will use saved model data, unless the `data/model` directory is deleted.
