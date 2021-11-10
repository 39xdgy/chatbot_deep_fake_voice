# Chatbot



In the first month, we tested how to train the word2vec model, and tested the word2vec pre-trained model trained on a large dataset.



The data we use to train the word2vec model is Penn Treebank (PTB), which is a commonly used small corpus. It is sampled from "Wall Street Journal" articles, including training set, validation set and test set. This project selected 2,499 stories from a three year Wall Street Journal (WSJ) collection of 98,732 stories for syntactic annotation.



In the preprocessing stage, we performed the following operations on the data. The first step is build a token-index pair for each word. In order to simplify the calculation, we only keep words that appear at least 5 times in the data set. Then map the words to integer indexes. After that, we should deal with the high-frequency words. Some high-frequency words generally appear in text data, such as "the", "a" and "in" in English. Generally speaking, in a background window, a word (such as "chip") and a lower frequency word (such as "microprocessor") appear at the same time than a higher frequency word (such as "the") appears at the same time to train the word embedding model More beneficial. Therefore, words can be sub-sampled when training the word embedding model. Specifically, each indexed word in the data set will have a certain probability to be discarded, and the higher the frequency of occurrence, the higher the probability of being discarded.



Training a neural network means inputting training samples and constantly adjusting the weights of neurons, so as to continuously improve the accurate prediction of the target. Whenever the neural network is trained with a training sample, its weight will be adjusted once. The word2vec model has a large-scale weight matrix, and training these data consumes computational resources and is very slow. So the calculation needs to be adjusted. In the training process of the word2vec model, we do not perform calculations on all the data. For each data, negative sampling is performed in the data set, and these samples are used to calculate the loss function. According to the definition of the loss function in negative sampling, we can use the binary cross-entropy loss function.



To train the word2vec model, we use the Skip-gram model. The Skip-gram model assumes that the words surrounding the text sequence are generated based on a certain word. After training the word embedding model, we can express the semantic similarity between words based on the cosine similarity of the two word vectors. According to this model, we can check the similarity between different words and find synonyms or analogies.



We can use our own corpus for training, or we can directly use models that have been trained on large datasets. We tested the pre-trained model provided by GloVe word embedding. GloVe contains the following corpora: 

['glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

After completing these, we can proceed to the next step. Now, we are using lstm to test the IMDb dataset and build a seq2seq model on this basis for translation and dialogue tasks.

