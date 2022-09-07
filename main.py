import nltk

nltk.download('twitter_samples')

# Before using a tokenizer in NLTK, you need to download an additional resource, punkt. The punkt module 
# is a pre-trained model that helps you tokenize words and sentences. For instance, this model knows that 
# a name may contain a period (like “S. Daityari”) and the presence of this period in a sentence does not 
# necessarily end it.
nltk.download('punkt')

from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json') # 5000 tweets with positive sentiments
negative_tweets = twitter_samples.strings('negative_tweets.json') # 5000 tweets with negative sentiments
text = twitter_samples.strings('tweets.20150430-223406.json') # 20000 tweets with no sentiments

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
print(tweet_tokens[0])
