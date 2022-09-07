import nltk
# nltk.download('twitter_samples')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk import FreqDist
import re, string

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
    print(freq_dist_pos.most_common(10))
    