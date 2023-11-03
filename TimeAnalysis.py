import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

time_df= pd.read_csv('RemainingTweets.csv')
time_df = time_df.drop(['User','Hashtag','Text', 'Transformer Classify', 'TextBlob Classify'], axis = 1)

graphTime = [item.split(" ")[1] for item in time_df['Date'].values]
graphHour = [item.split(":")[0] for item in graphTime]
time_df["Graph Time"] = graphHour
time_df=time_df.groupby("Vader Classify")


neg_df = time_df.get_group('negative')
pos_df = time_df.get_group('positive')
neutral_df = time_df.get_group('neutral')

neg_counts = neg_df.groupby("Graph Time")["Vader Classify"].count()
graph_df = neg_counts.to_frame()

graph_df.columns = ["Negative Count"]

pos_counts = pos_df.groupby("Graph Time")["Vader Classify"].count()
graph_df["Positive Count"] = pos_counts.to_frame()

neutral_counts = neutral_df.groupby("Graph Time")["Vader Classify"].count()
graph_df["Neutral Count"] = neutral_counts.to_frame()


ax = graph_df.plot.bar(rot=0, stacked=True, title= 'Vader Sentiment per Hour')
plt.xlabel("Hour (Recorded in UTC)")
plt.ylabel("Number of Tweets")
plt.savefig('TimeAnalysisNoOutliers.jpg')
