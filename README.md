# Aspect-based-Sentiment-Analysis
> Implementation of some aspect-based sentiment analysis models;
based on car dataset;
3 classes

### Models

- [ATAE-LSTM(Attention-based LSTM with Aspect Embedding)](http://aclweb.org/anthology/D16-1058)  
Attention-based LSTM for Aspect-level Sentiment Classification

- [TSA(A Siamese Bidirectional LSTM with context-aware attention)](https://www.aclweb.org/anthology/S17-2126/)
DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis


### Data Analysis

|            item                | value  |
|--------------------------------|--------|
| training set                   | 12813  |
| valid/dev set                  | 1602   |
| test set                       | 1602   |
| char_vocab                     | 2378   |
| all_char_vocab                 | 2379   |
| aspect                         | 20     |
| aspect_text_char_vocab         | 69     |
| char_max_len                   | 127    |
| < char_len = 0.991             | 110    |
| aspect_text_char_max_len       |   19   |
| < aspect_text_char_len = 0.978 |   18   |

### outputs(performance)

+ random

| model   |  acc(on test)    | acc(on dev)  | macro-f1(test)|macro-f1(dev)|
|---------|------------------|--------------|-------------|---------------|
|atae_lstm|  0.6536          |  0.6685      |   0.5952    |     0.6143    |
|tsa      |  0.6654          |  0.6816      |   0.6194    |     0.6381    |

+ word2vector

| model   |  acc(on test)    | acc(on dev)  | macro-f1(test)|macro-f1(dev)|
|---------|------------------|--------------|-------------|---------------|
|atae_lstm|  0.6704          |  0.6685      |   0.6351    |     0.6347    |
|tsa      |  0.6710          |  0.6792      |   0.6296    |     0.6445    |

+ glove

| model   |  acc(on test)    | acc(on dev)  | macro-f1(test)|macro-f1(dev)|
|---------|------------------|--------------|-------------|---------------|
|atae_lstm|  0.6461          |  0.6754      |   0.5863    |     0.6226    |
|tsa      |  0.6567          |  0.6798      |   0.6198    |     0.6461    |

> screenshots --> outputs file

### str tree(run codes)

    .
    ├── ckpt
    |   ├── car    saved model files
    |   └── others
    ├── data
    │   ├── car    preprocessed data files
    |   └── others
    ├── xxx.csv    csv files of pre & rl labels
    ├── config.py
    ├── data_loader.py
    ├── layers.py
    ├── models.py
    ├── preprocess.py
    ├── train.py
    └── utils.py


