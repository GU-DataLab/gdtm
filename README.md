# Georgetown DataLab Topic Modeling package (gdtm)
A Python Package containing wrappers and classes for topic models, including Topic-Noise Discriminator (TND), (Noiseless LDA) NLDA, Guided Topic-Noise Model (GTM), dynamic topic-noise models, and Percolation-based Topic Model (PTM).

The documentation for `gdtm` can be found [here](https://gu-datalab.github.io/gdtm/).

### Requirements and setup
to install `gdtm`, run  
`pip install gdtm`

gdtm requires a version of Mallet LDA and/or the Mallet implementations of the other topic models in order to run.  
You can download our stripped-down versions of Mallet LDA and the other topic model implementations in this [repository](https://github.com/GU-DataLab/topic-noise-models-source).
If you want the full Mallet Java package, you can find it [here](http://mallet.cs.umass.edu).

### Using gdtm
There are two ways to use this package.  The first is to use a topic model class, like `TND`. The second is to use a wrapper (based on the now deprecated Gensim Mallet wrapper), like `TNDMallet`.  The class calls the wrapper underneath, but classes add a layer that is easier to use and interact with than that of the wrapper itself.  

Wrappers would be useful when you are developing your own Mallet-based models, or when you want full control over the flow of data.  
Otherwise, classes are easier to deal with. Ensembled topic models, like NLDA and its variants, are only available through a class, which calls multiple wrappers and does the ensembling in a class function.


### Data set loading and structure
Data sets can be loaded in whatever way is most convenient, as long as they are passed into the `gdtm` functions in the same format. The format of a data set is a list of documents, where each document is a list of words.
```python
dataset = [['doc1_word1', 'doc1, word2', ...], ['doc2_word1', 'doc2_word2', ...], ...]
```

We provide functions for loading data sets into our format from files.  The first loads a "flat" data set.
```python
from src.gdtm.helpers.common import load_flat_dataset

path_to_data = 'path/to/data/tweets.csv'
dataset = load_flat_dataset(path_to_data, delimiter=' ')
```

We also provide functions for loading and formatting temporal data sets.  The following takes a file where each row is a dated document (example format: `date\tw1,w2,w3,w4`, where `\t` is a tab), loads in the data, splits the data by date into epochs, saves the "split" data set, and then loads the split data set.

```python
from src.gdtm.helpers.common import load_dated_dataset, load_split_dataset, split_dataset_by_date, save_split_dataset, month

path_to_data = 'path/to/data/'
dataset = load_dated_dataset(path='{}dated_tweets.csv'.format(path_to_data), date_delimiter='\t', doc_delimiter=',')

split_dataset = split_dataset_by_date(dataset, epoch_function=month)
num_time_periods = len(split_dataset.keys())
save_split_dataset(path=path_to_data, file_name='split_dataset', dataset=dataset, delimiter=' ')

loaded_dataset = load_split_dataset(path=path_to_data, file_name='split_dataset', 
                                    num_time_periods=num_time_periods, delimiter=' ')
```


### Using a class-based topic model
```python
from src.gdtm.models import NLDA

# Set these paths to the path where you saved the Mallet implementation of each model, plus bin/mallet
tnd_path = 'path/to/mallet-tnd/bin/mallet'
lda_path = 'path/to/mallet-lda/bin/mallet'

# We pass in the paths to the java code along with the data set and whatever parameters we want to set
model = NLDA(dataset=dataset, mallet_tnd_path=tnd_path, mallet_lda_path=lda_path, 
             tnd_k=30, lda_k=30, nlda_phi=10, top_words=20, save_path='path/to/results/')

topics = model.get_topics()
noise = model.get_noise_distribution()
```

### Using a wrapper-based topic model
```python
from gensim import corpora
from src.gdtm.wrappers import TNDMallet

# Set the path to the path where you saved the Mallet implementation of the model, plus bin/mallet
tnd_path = 'path/to/mallet-tnd/bin/mallet'

# Format the data set for consumption by the wrapper (this is done automatically in class-based models)
dictionary = corpora.Dictionary(dataset)
dictionary.filter_extremes()
corpus = [dictionary.doc2bow(doc) for doc in dataset]
# Pass in the path to the java code along with the data set and parameters
model = TNDMallet(tnd_path, corpus, num_topics=30, id2word=dictionary, 
                  skew=25, noise_words_max=200, iterations=1000)

topics = model.get_topics()
noise = model.load_noise_dist()
```

### Acknowledgements
First and foremost, we would like to thank the creators of Gensim and Mallet for writing such incredible code that made our lives so much easier when it came to implementing our own fast, accurate topic models.

This work was supported by the Massive Data Institute at Georgetown University, and by the Nation Science Foundation.  We would like to thank our funders.

### Referencing TND and NLDA
```
Churchill, Rob and Singh, Lisa. 2021. Topic-Noise Models: Modeling Topic and Noise Distributions in Social Media Post Collections. International Conference on Data Mining (ICDM).
```

```bibtex 
@inproceedings{churchill2021tnd,
author = {Churchill, Rob and Singh, Lisa},
title = {Topic-Noise Models: Modeling Topic and Noise Distributions in Social Media Post Collections},
booktitle = {International Conference on Data Mining (ICDM)},
year = {2021},
}
```

### Referencing GTM
```
Churchill, Rob and Singh, Lisa. 2022. A Guided Topic-Noise Model for Short Texts. The Web Conference (WWW).
```

```bibtex 
@inproceedings{churchill2021gtm,
author = {Churchill, Rob and Singh, Lisa},
title = {A Guided Topic-Noise Model for Short Texts},
booktitle = {The Web Conference (WWW)},
year = {2022},
}
```

### Referencing the Dynamic Topic-Noise Models
```
Churchill, Rob and Singh, Lisa. 2022. Dynamic Topic-Noise Models for Social Media. Pacific-Asia Conference on Knowledge Discovery
and Data Mining (PAKDD).
```

```bibtex 
@inproceedings{churchill2021gtm,
author = {Churchill, Rob and Singh, Lisa},
title = {Dynamic Topic-Noise Models for Social Media},
booktitle = {Pacific-Asia Conference on Knowledge Discovery
and Data Mining (PAKDD)},
year = {2022},
}
```

### References
1. A. K. McCallum, “Mallet: A machine learning for language toolkit.” 2002.  
2. David M. Blei, Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." Journal of Machine Learning Research, 3:993–1022, 3 2003.
3. P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, "Enriching Word Vectors with Subword Information." In: Transactions of the Association for Computational Linguistics (TACL). 2017.  
4. R. Rehurek, P. Sojka, "Gensim–python framework for vector space modelling." 2011.  
5. R. Churchill, L. Singh, "Percolation-based topic modeling for tweets." In: KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM). 2020.  
6. R. Churchill, L. Singh, "Topic-noise models: Modeling topic and noise distributions in social media post collections." In: International Conference on Data Mining (ICDM). 2021.  
7. R. Churchill, L. Singh, "Dynamic Topic-Noise Models for Social Media." In: Pacific-Asia Conference on Knowledge Discovery
and Data Mining (PAKDD). 2022.  
8. R. Churchill, L. Singh, "A Guided Topic-Noise Model for Short Texts." In: The Web Conference (WWW). 2022.
