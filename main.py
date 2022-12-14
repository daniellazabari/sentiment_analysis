import nltk
# nltk.download('twitter_samples')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string
import random

def remove_noise(tweet_tokens, stop_words=()):
    '''
    This function uses regexp to remove from the data: hyperlinkes, twitter handles in replies and 
    punctuation and special characters.
    Then, the function uses lemmatization to normalize the data. 
    '''
    cleaned_tokens = []
    # Get the position tag of each token of a tweet
    for token, tag in pos_tag(tweet_tokens):
        # Remove noise from the data
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token) # Remove hyperlinkes 
        token = re.sub("(@[A-Za-z0-9_]+)", "", token) # Remove twittwer handles in replies
        
        # Normalize the data
        # If the tag starts with NN, the token is assigned as a noun
        if tag.startswith('NN'):
            pos = 'n'
        # If the tag starts with VB, the token is assigned as a verb
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    
    return cleaned_tokens

def get_all_words(cleaned_tokens):
    '''
    A Generator function that takes a list of tweets as an argument to provide a list of words in all of
    the tweet tokens joined. 
    '''
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens):
    '''
    A Generator function that converts the tweets from a list of cleaned tokens to dictionaries with 
    tokens as keys and True as values.
    '''
    for tweet_tokens in cleaned_tokens:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":
    # Tokenize the data
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

    stop_words = stopwords.words('english')
    
    # Cleaned the sample tweets
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens = []
    negative_cleaned_tokens = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens.append(remove_noise(tokens, stop_words))
    
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens)
    freq_dist_pos = FreqDist(all_pos_words)

    # Prepaaring data for the model
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens)
    
    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]
    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    # Build the model
    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    # Test the model
    custom_tweet = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print(classifier.classify(dict([token, True] for token in custom_tokens)))

