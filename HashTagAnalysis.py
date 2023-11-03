import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import and clean the recorded data
hashtag_df= pd.read_csv('RemainingTweets.csv')
hashtag_df = hashtag_df.drop(['User', 'Date','Text'], axis = 1)

#Group by the hashtags and make new dataframes
hashtag_df = hashtag_df.groupby('Hashtag')
politics_df = hashtag_df.get_group('#politics')
democrat_df = hashtag_df.get_group('#democrat')
republican_df = hashtag_df.get_group('#republican')

#For '#Politics' Get counts of each form of analysis and type of sentiment
pol_count_df = politics_df["Vader Classify"].value_counts(sort=True).to_frame()
pol_count_df["Transformer Classify"] = politics_df["Transformer Classify"].value_counts(sort=True).to_frame()
pol_count_df["Transformer Classify"] = pol_count_df["Transformer Classify"].fillna(0)
pol_count_df["Transformer Classify"] = pol_count_df["Transformer Classify"].astype(int)
pol_count_df["TextBlob Classify"] = politics_df["TextBlob Classify"].value_counts(sort=True).to_frame()
pol_count_df.index=['Negative', "Positive", "Neutral"]

#Make a #Politics table of sentiment analysis
pol_count_figure = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Sentiment Breakdown of Tweets with #Politics', 
             fontweight ="bold") 
pol_count_table = ax.table(cellText= pol_count_df.values, rowColours=['lightgrey']*pol_count_df.shape[0],
 colColours=['lightgrey']*pol_count_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels= pol_count_df.columns, rowLabels = pol_count_df.index)
#plt.savefig('PoliticsTableNoOutliers.jpg')
#plt.show()

#Make #Politics bar graph
ax = pol_count_df.plot.bar(rot=0, title= 'Sentiment Breakdown of Tweets with #Politics')
#plt.savefig('PoliticsGraphNoOutliers.jpg')
#plt.show()

#For '#Democrat' Get counts of each form of analysis and type of sentiment
dem_count_df = democrat_df["Vader Classify"].value_counts(sort=True).to_frame()
dem_count_df["Transformer Classify"] = democrat_df["Transformer Classify"].value_counts(sort=True).to_frame()
dem_count_df["Transformer Classify"] = dem_count_df["Transformer Classify"].fillna(0)
dem_count_df["Transformer Classify"] = dem_count_df["Transformer Classify"].astype(int)
dem_count_df["TextBlob Classify"] = democrat_df["TextBlob Classify"].value_counts(sort=True).to_frame()
dem_count_df.index=['Negative', "Positive", "Neutral"]
print(dem_count_df.head(10))
#Make a '#Democrat' table of sentiment analysis
dem_count_figure = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Sentiment Breakdown of Tweets with #Democrat', 
             fontweight ="bold") 
dem_count_table = ax.table(cellText= dem_count_df.values, rowColours=['lightgrey']*dem_count_df.shape[0], colColours=['lightgrey']*dem_count_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels= dem_count_df.columns, rowLabels = dem_count_df.index)
#plt.savefig('DemocratTableNoOutliers.jpg')
#plt.show()

#Make a '#Democrat' bar graph
ax = dem_count_df.plot.bar(rot=0, title= 'Sentiment Breakdown of Tweets with #Democrat')
#plt.savefig('DemocratGraphNoOutliers.jpg')
#plt.show()


#For '#Republican' Get counts of each form of analysis and type of sentiment
rep_count_df = republican_df["Vader Classify"].value_counts(sort=True).to_frame()
rep_count_df["Transformer Classify"] = republican_df["Transformer Classify"].value_counts(sort=True).to_frame()
rep_count_df["Transformer Classify"] = rep_count_df["Transformer Classify"].fillna(0)
rep_count_df["Transformer Classify"] = rep_count_df["Transformer Classify"].astype(int)
rep_count_df["TextBlob Classify"] = republican_df["TextBlob Classify"].value_counts(sort=True).to_frame()
rep_count_df.index=['Negative', "Positive", "Neutral"]

#Make a '#Republican' table of sentiment analysis
rep_count_figure = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Sentiment Breakdown of Tweets with #Republican', 
             fontweight ="bold") 
rep_count_table = ax.table(cellText= rep_count_df.values, rowColours=['lightgrey']*rep_count_df.shape[0], colColours=['lightgrey']*rep_count_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels= rep_count_df.columns, rowLabels = rep_count_df.index)
#plt.savefig('RepublicanTableNoOutliers.jpg')
#plt.show()

#Make a '#Republican' bar graph
ax = rep_count_df.plot.bar(rot=0, title= 'Sentiment Breakdown of Tweets with #Republican')
#plt.savefig('RepublicanGraphNoOutliers.jpg')
#plt.show()

