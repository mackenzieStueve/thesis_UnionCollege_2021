import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

outlier_df= pd.read_csv('outlierTweets.csv')
outlier_df = outlier_df.drop(['Unnamed: 0'], axis = 1)


question_outlier= outlier_df['Text'].str.count("\\?").to_frame()
question_outlier = question_outlier.rename(columns={"Text": "Num Tweets"})
question_count = question_outlier["Num Tweets"].value_counts(sort=True).to_frame()
#print(question_count)


count_df = outlier_df["Vader Classify"].value_counts(sort=True).to_frame()
count_df["Transformer Classify"] = outlier_df["Transformer Classify"].value_counts(sort=True).to_frame()
count_df["Transformer Classify"] = count_df["Transformer Classify"].fillna(0)
count_df["Transformer Classify"] = count_df["Transformer Classify"].astype(int)
count_df["TextBlob Classify"] = outlier_df["TextBlob Classify"].value_counts(sort=True).to_frame()
count_df.index=["Positive", "Neutral", 'Negative']


count_figure = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Recorded Sentiment By Analysis Method', 
             fontweight ="bold") 
count_table = ax.table(cellText=count_df.values, rowColours=['lightgrey']*count_df.shape[0], colColours=['lightgrey']*count_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels=count_df.columns, rowLabels = count_df.index)
plt.show()