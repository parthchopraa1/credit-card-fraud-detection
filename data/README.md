# Dataset — Credit Card Fraud Detection (ULB / Worldline)

## File

`creditcard.csv` — 284,807 transactions by European cardholders over 2 days in
September 2013. 492 transactions (0.1727%) are fraudulent.

## Source

The canonical source is the Kaggle dataset **mlg-ulb/creditcardfraud**:

- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This copy was downloaded from the public TensorFlow-hosted mirror of the same
file (no login required):

- https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Verified locally: 284,807 rows, 492 frauds, columns
`Time, V1–V28, Amount, Class` — identical to the Kaggle release.

## Columns

| Column | Meaning |
| ------ | ------- |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1`–`V28` | PCA-transformed features (originals withheld for confidentiality) |
| `Amount` | Transaction amount |
| `Class` | Target — 0 = legitimate, 1 = fraud |

## Licence / citation

The dataset is released by the ULB Machine Learning Group (Worldline) for
research use. If you use it, cite:

> Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> IEEE Symposium Series on Computational Intelligence (SSCI).

Note: the file is ~144 MB, so it is **not** committed to version control.
Download it from one of the links above and place it at `data/creditcard.csv`.
