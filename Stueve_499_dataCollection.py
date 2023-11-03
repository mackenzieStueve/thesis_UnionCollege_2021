#LAST TIME RUN: 2021-02-23 23:53:12.830128


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tweepy
import re
import os
import json

#import the necessary sentiment analysis tools and open up the calls
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
vaderAnalyzer = SentimentIntensityAnalyzer()
from transformers import pipeline
transformerSentiment = pipeline('sentiment-analysis')

#load in specific twitter info from developer account
consumer_key = "mmLHllZ6yncexXHfV05LMF53L"
consumer_secret = "09GG9Y6XCgSN1Eo1Ujq5maGz0WCZNqRla96omtvLPof7LCUZP8"
access_key = "1318200756139417601-a4OZeyHeAlA0CpIWmqKy8tnNAkN4z6"
access_secret = "tf6rbLQnLc67vf1OjJMZX2npmIkTqxrQeM5zpXdwoHm6I"

#gain access to the twitter api using the keys
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#list of hashtags searching
hashtag_searches = "#politics", "#democrat", "#republican"

#import the last the program was run, this is to check for strictly new tweets.
date_since= "2021-02-22"
exact_datetime = "2021-02-23 23:53:13"
exact_datetime_since = datetime.datetime.strptime(exact_datetime, "%Y-%m-%d %H:%M:%S")
#open empty list in order to store the tweet information as toubles in the list
originalTweets = []
#iterate through each of the hashtags
for hashtag in hashtag_searches:
    #the line below calls to the twitter API and stores all items for that specific hashtag in allTweets
    allTweets = tweepy.Cursor(api.search, q = hashtag+" -filter:retweets AND -filter:replies AND -filter:quote AND -filter:links AND -filter:media",
                                 lang = "en",tweet_mode = "extended", since= date_since).items(1000)
    for tweet in allTweets:
        #checks if the tweet was created after the specific time of the last tweet, if so stores in tweet list
        if tweet.created_at > exact_datetime_since:
            originalTweets.append({"User": tweet.user.screen_name,
                                "Hashtag": hashtag,
                                "Date": tweet.created_at,
                                "Text":tweet.full_text
                        })

#turn the saved data into a dataframe
politics_df = pd.DataFrame.from_dict(originalTweets)

sentiment_scores = []

#iterate through the text of the tweets
for i in range(politics_df.shape[0]):
    #get the vader compound score
    vaderCompound = vaderAnalyzer.polarity_scores(politics_df['Text'][i])["compound"]
    #get the hugging face transformer sentiment
    tansformerResult = transformerSentiment(politics_df['Text'][i])
    #get the textblob polarity score
    textBlobReturn = TextBlob(politics_df['Text'][i])
    textBlobValue = textBlobReturn.sentiment.polarity
    #Assign the sentiment based on given interpretation values
    if vaderCompound >= .05:
        vaderSentiment = "positive"
    if vaderCompound <= -.05:
        vaderSentiment = "negative"
    if (vaderCompound > -.05) and (vaderCompound < .05):
        vaderSentiment = "neutral"
    #Use the textBlob polarity score to assign sentiment
    if textBlobValue >= .01:
        textBlobSentiment = "positive"
    if textBlobValue <= -.01:
        textBlobSentiment = "negative"
    if (textBlobValue > -.01) and (textBlobValue < .01):
        textBlobSentiment = "neutral"
    #Store the polarity into the appropriate collumn header
    sentiment_scores.append({"Vader Compound": vaderCompound,
                                        "Vader Classify": vaderSentiment,
                                        "Transformer Score": tansformerResult[0]['score'],
                                        "Transformer Classify": tansformerResult[0]['label'].lower(),
                                        "TextBlob Polarity": textBlobValue,
                                        "TextBlob Classify": textBlobSentiment})
#turn these into a dataframe
vader_sentiments_score = pd.DataFrame.from_dict(sentiment_scores)
politics_df = politics_df.join(vader_sentiments_score)


#print out the run time in order to know when the tweets were stored.
print("Time run: ", datetime.datetime.utcnow())

#convert the data frame into a csv file to store the results.
politics_df.to_csv('fullSetPoliticalTweets.csv', mode='a', index=False, header=False)