import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import and clean the recorded data
comparison_df= pd.read_csv('fullSetPoliticalTweets.csv')
comparison_df = comparison_df.drop(['Vader Compound', 'Transformer Confidence', 'TextBlob Polarity'], axis = 1)

#Create counts of each sentimiment by sentiment method
count_df = comparison_df["Vader Classify"].value_counts(sort=True).to_frame()
count_df["Transformer Classify"] = comparison_df["Transformer Classify"].value_counts(sort=True).to_frame()
count_df["Transformer Classify"] = count_df["Transformer Classify"].fillna(0)
count_df["Transformer Classify"] = count_df["Transformer Classify"].astype(int)
count_df["TextBlob Classify"] = comparison_df["TextBlob Classify"].value_counts(sort=True).to_frame()
count_df.index=['Negative', "Positive", "Neutral"]

#Create and export a table with the above info
count_figure = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Recorded Sentiment By Analysis Method', 
             fontweight ="bold") 
count_table = ax.table(cellText=count_df.values, rowColours=['lightgrey']*count_df.shape[0], colColours=['lightgrey']*count_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels=count_df.columns, rowLabels = count_df.index)
#plt.savefig('AllTweetsCountTable.jpg')

#Make Bar Graph and Export
ax = count_df.plot.bar(rot=0, title= 'Recorded Sentiment By Analysis Method')
#plt.savefig('AllTweetsCountGraph.jpg')


#Record the comparison of different types of analysis
vader_transformer_compare = np.where(comparison_df["Vader Classify"] == comparison_df["Transformer Classify"], "Agree", "Disagree")
comparison_df["Vader & Transformer Comparison"] = vader_transformer_compare

vader_textblob_compare = np.where(comparison_df["Vader Classify"] == comparison_df["TextBlob Classify"], "Agree", "Disagree")
comparison_df["Vader & TextBlob Comparison"] = vader_textblob_compare

transformer_textblob_compare = np.where(comparison_df["Transformer Classify"] == comparison_df["TextBlob Classify"], "Agree", "Disagree")
comparison_df["Transformer & TextBlob Comparison"] = transformer_textblob_compare




#Record how frequently all the sentiments agree
all_equal_compare = np.where((comparison_df["Vader & Transformer Comparison"] == comparison_df["Vader & TextBlob Comparison"])
                           & (comparison_df["Vader & Transformer Comparison"] == "Agree") , "Agree", "Disagree")
comparison_df["Total Agreement"] = all_equal_compare
agreement_df = comparison_df["Total Agreement"].value_counts(sort=True).to_frame()
get_text_df = comparison_df.groupby("Total Agreement")
agreement_text_df = get_text_df.get_group("Agree")
agreement_text_df = agreement_text_df.drop(['Vader & Transformer Comparison', 'Vader & TextBlob Comparison',
 'Transformer & TextBlob Comparison', 'Total Agreement'], axis = 1)
agreement_text_df.to_csv('TotalAgreementTweets.csv', mode='a', index=False)#, header=False)
all_agree_count = agreement_df.loc["Agree",:]["Total Agreement"]
print(all_agree_count)


#Record how frequently all the sentiments disagree
all_different_compare = np.where((comparison_df["Vader & Transformer Comparison"] == "Disagree") & 
                           (comparison_df["Vader & TextBlob Comparison"] == "Disagree") &
                           (comparison_df["Transformer & TextBlob Comparison"] == "Disagree"), "Agree", "Disagree")
comparison_df["Total Disagreement"] = all_different_compare
disagreement_df = comparison_df["Total Disagreement"].value_counts(sort=True).to_frame()

all_disagree_count = disagreement_df.loc["Agree",:]["Total Disagreement"]
print(all_disagree_count)


#Export this data into a new CSV file for further analysis
#comparison_df.to_csv('analysisDataPoliticalTweets.csv', mode='a', index=False)

#Count the number of agreeing vs disagreeing sentiment between method
similarity_df = comparison_df['Vader & Transformer Comparison'].value_counts(sort=True).to_frame()
similarity_df.columns=['Vader & Transformer']
similarity_df['Vader & TextBlob'] = comparison_df['Vader & TextBlob Comparison'].value_counts(sort=True).to_frame()
similarity_df['Transformer & TextBlob'] = comparison_df['Transformer & TextBlob Comparison'].value_counts(sort=True).to_frame()

chartValues = [[2658, similarity_df.loc["Agree",:]["Vader & Transformer"], similarity_df.loc["Agree",:]["Vader & TextBlob"]],
                [similarity_df.loc["Agree",:]["Vader & Transformer"], 2658, similarity_df.loc["Agree",:]["Transformer & TextBlob"]],
                [similarity_df.loc["Agree",:]["Vader & TextBlob"], similarity_df.loc["Agree",:]["Transformer & TextBlob"], 2658]]


chartLabels= ["Vader", "Transformer", "TextBlob"]
similarity_fig = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Comparing Similarity Between Sentiment Analysis Methods', 
             fontweight ="bold") 
similarity_table = ax.table(cellText=chartValues, rowColours=['lightgrey']*similarity_df.shape[1],
 colColours=['lightgrey']*similarity_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels=chartLabels, rowLabels = chartLabels)
#plt.savefig('MethodAgreeanceTable.jpg')



#Create the table in order to record the agreeing and disagreeing values & export it
similarity_fig = plt.figure(figsize=(9,2))
ax=plt.subplot(111)
ax.axis('off') 
ax.set_title('Comparing Similarity Between Sentiment Analysis Methods', 
             fontweight ="bold") 
similarity_table = ax.table(cellText=similarity_df.values, rowColours=['lightgrey']*similarity_df.shape[0],
 colColours=['lightgrey']*similarity_df.shape[1],
 bbox=[0, 0, 1, 1], colLabels=similarity_df.columns, rowLabels = similarity_df.index)
#plt.savefig('AllTweetsSimilarityTable.jpg')

