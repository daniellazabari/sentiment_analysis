import nltk

# nltk.download('twitter_samples')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples

# Tokenize the data
positive_tweets = twitter_samples.strings('positive_tweets.json') # 5000 tweets with positive sentiments
negative_tweets = twitter_samples.strings('negative_tweets.json') # 5000 tweets with negative sentiments
text = twitter_samples.strings('tweets.20150430-223406.json') # 20000 tweets with no sentiments
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

# Normalize the data using lemmatization, which normalized a word with the context of vocabulary and
# morphological analysis of words in text.
# Before running a lemmatizer, you need to determine the context for each word in the text using a 
# tagging algorithm.
def lemmatize_sentence(tokens):
    '''
    In this function, we first generate the tags for each token in the text, and 
    then lemmatize each word using the tag. 
    '''
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    # Get the position tag of each token of a tweet
    for word, tag in pos_tag(tokens):
        # If the tag starts with NN, the token is assigned as a noun
        if tag.startswith('NN'):
            pos = 'n'
        # If the tag starts with VB, the token is assigned as a verb
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
