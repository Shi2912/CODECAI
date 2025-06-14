import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Load Dataset

df = pd.read_csv('Tweets.csv')

# Keep only relevant columns
df = df[['text', 'airline_sentiment']]
df.rename(columns={'text': 'tweets', 'airline_sentiment': 'original_sentiment'}, inplace=True)


# Clean Tweets

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)           # remove URLs
    tweet = re.sub(r"@\w+", "", tweet)              # remove mentions
    tweet = re.sub(r"#\w+", "", tweet)              # remove hashtags
    tweet = re.sub(r"[^A-Za-z\s]", "", tweet)       # remove special characters
    tweet = tweet.lower()                           # convert to lowercase
    tweet = " ".join(word for word in tweet.split() if word not in stop_words)  # remove stopwords
    return tweet

df['cleaned'] = df['tweets'].apply(clean_tweet)


# Sentiment Classification

def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['predicted_sentiment'] = df['cleaned'].apply(get_sentiment)


# Show Some Results

print("\n📊 First 10 Predicted Results:\n")
print(df[['tweets', 'original_sentiment', 'predicted_sentiment']].head(10))


# Visualization

plt.figure(figsize=(8, 5))
df['predicted_sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Analysis of Tweets (Predicted)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Accuracy Comparison 
from sklearn.metrics import classification_report

# Only compare where original sentiment is among 3 classes
valid_df = df[df['original_sentiment'].isin(['positive', 'negative', 'neutral'])]

print("\n📋 Classification Report (vs Original Labels):\n")
print(classification_report(valid_df['original_sentiment'].str.capitalize(),
                            valid_df['predicted_sentiment']))


# Final Check – Test Custom Tweet

print("\n🧪 Sentiment Prediction Test")
user_tweet = input("Type a tweet to analyze its sentiment: ")

cleaned_input = clean_tweet(user_tweet)
predicted = get_sentiment(cleaned_input)

print(f"\nPrediction: The tweet is **{predicted.upper()}**.")
