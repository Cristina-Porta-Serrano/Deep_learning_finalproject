# Deep_learning_finalproject
Text sentiment analysis: drug reviews

## State of art 
The dataset was originally published on the UCI Machine Learning repository.
The dataset provides patient reviews on specific medications along with related conditions and a 10-star patient rating that reflects overall patient satisfaction. Data was obtained by crawling online pharmaceutical review sites.

Attribute information:
1. drugName (categorical): name of the drug.
2. condition (categorical): name of the condition.
3. review (text): patient review.
4. rating (numeric): 10-star patient rating.
5. date (date): revision entry date.
6. usefulCount (numeric): number of users who found the review useful.

## Results
First, I used traditional ML algorithms:
- Random Forest
- LGBM
- Cat Boost

Then, I used algorithms from DL:
- A neural network with customized LSTM
- A pretrained neural network – GloVe

Before training any model, we must vectorize the text since the algorithms can only work with numerical data. We use the Tfidf (Term frequency – Inverse document frequency), (that is, the frequency of occurrence of the term in the collection of documents), it is a numerical measure that expresses how relevant a word is for a document in a collection. This measure is often used as a weighting factor in information retrieval and text mining. The tf-idf value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the document collection, which allows for handling the fact that some words are generally more common than others.

Having unbalanced data, my idea was to start with a Balanced Random Forest algorithm but I had some problems with Bootstrap that didn't let me implement it, so I finally opted for a normal Random Forest (which has finally been the model that has given the best results).

Considering the capabilities of the Light Gradient Boosting Machine (LGBM) in handling unbalanced data, it was my second choice. It worked almost as well as RF, in fact.

Initially, I believed that text-specific neural networks like LSTM with embeddings would perform better than traditional ML algorithms. After testing several networks with different number of layers, dropouts, loss-functions... the result has been somewhat worse than with RF and LGBM.

Lastly, I tested a neural network with pre-trained embeddings (GloVe). GloVe, Global Vectors, is a model for the representation of distributed words. The model is an unsupervised learning algorithm to obtain vector representations of words. This is achieved by assigning words to significant space where the distance between words is related to semantic similarity. It is the second model with better predictions that I have tried.

<img
  src="C:\Users\Cristina\OneDrive\Documentos\Curs SEPE\Deep_learning_finalproject/taula_resultats.jpg"
  alt="Alt text"
  title="Taula resultats"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  
## Conclusions
As it is not a complex data set and has many features that can help with prediction, a traditional ML algorithm has worked better for us than a neural network.
We have achieved better performance with a fairly small dataframe (a bit over 1,000 observations) than with the general dataset (of almost 200,000 observations, from which we had taken a sample of 9,000).

## Improvements
- First of all, a more exhaustive analysis of the first dataset would have to be done to clean it perfectly. I noticed that there were some fields in the "condition" feature that were not null, therefore not seen in the search for NaNs, but that they were incorrect, but there was not enough time to fix it since the same drug was used for different conditions.

- When vectorizing the words, word pairs could be used instead of single words to improve model training with more context, but it was very computationally slow.
Surely, applying a Balanced RF or having done undersampling of the majority class would help the predictions.

- Also, I would like to have used Transformers (such as BERT) to improve neural network predictions, but everything I tried gave me errors and I didn't know why.

## Requirements
This notebook requires a Python 3.6 or newer version.

To run this notebook you must have installed the following libraries:

    pip install pandas
    pip install numpy
    pip install matplotlib
    pip install tensorflow
    pip install -U scikit-learn
    pip install wordcloud
    pip install TextBlob
    pip install vaderSentiment
