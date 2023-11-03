import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hashtag_df= pd.read_csv('TotalAgreementTweets.csv')
hashtag_df = hashtag_df.drop(['Hashtag'], axis = 1)
print(len(hashtag_df.index))
hashtag_num= hashtag_df['Text'].str.count("\\#").to_frame()
hashtag_num=hashtag_num.rename(columns={"Text": "Num Tweets"})
hashtag_count = hashtag_num["Num Tweets"].value_counts(sort=True).to_frame()
print(hashtag_count)
print(hashtag_count["Num Tweets"].sum())





contains_pol= hashtag_df['Text'].str.contains('#politics', case=False)
contains_dem = hashtag_df['Text'].str.contains('#democrat', case=False)
contains_rep = hashtag_df['Text'].str.contains('#republican', case=False)

politics_df= hashtag_df[contains_pol]

democrat_df = hashtag_df[contains_dem]

republican_df = hashtag_df[contains_rep]


#For '#Politics' Get counts of each form of analysis and type of sentiment
pol_count_df = politics_df["Vader Classify"].value_counts(sort=True).to_frame()
pol_count_df = pol_count_df.rename(columns={"Vader Classify" : "#politics"})
pol_count_df["#democrat"] = democrat_df["Vader Classify"].value_counts(sort=True).to_frame()
pol_count_df["#republican"] = politics_df["Vader Classify"].value_counts(sort=True).to_frame()
pol_count_df.index=['Negative', "Positive"]
pol_count_df=pol_count_df.T
#print(pol_count_df.head())


#Make a #Politics table of sentiment analysis
ax = pol_count_df.plot.bar(rot=0, title= 'Sentiment Breakdown of Hashtags in Total Agreeance Tweets')
#plt.savefig('HashtagSentimentAgreeanceTweets.jpg')

