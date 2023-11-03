import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import and clean the recorded data
comparison_df= pd.read_csv('fullSetPoliticalTweets.csv')

vader_transformer_compare = np.where(comparison_df["Vader Classify"] == comparison_df["Transformer Classify"], "Agree", "Disagree")
comparison_df["Vader & Transformer Comparison"] = vader_transformer_compare

vader_textblob_compare = np.where(comparison_df["Vader Classify"] == comparison_df["TextBlob Classify"], "Agree", "Disagree")
comparison_df["Vader & TextBlob Comparison"] = vader_textblob_compare

transformer_textblob_compare = np.where(comparison_df["Transformer Classify"] == comparison_df["TextBlob Classify"], "Agree", "Disagree")
comparison_df["Transformer & TextBlob Comparison"] = transformer_textblob_compare


all_different_compare = np.where((comparison_df["Vader & Transformer Comparison"] == "Disagree") & 
                           (comparison_df["Vader & TextBlob Comparison"] == "Disagree") &
                           (comparison_df["Transformer & TextBlob Comparison"] == "Disagree"), "Agree", "Disagree")
comparison_df["Total Disagreement"] = all_different_compare
disagreement_df = comparison_df[comparison_df["Total Disagreement"]=="Agree"]

analysistweets = comparison_df[~comparison_df.Text.isin(disagreement_df.Text)]
analysistweets = analysistweets.drop(['Vader Compound', 'Transformer Confidence', 'TextBlob Polarity',
 'Vader & Transformer Comparison', 'Vader & TextBlob Comparison', 'Transformer & TextBlob Comparison', 'Total Disagreement'], axis = 1)
analysistweets.to_csv('RemainingTweets.csv', mode='a', index=False)#, header=False)

disagreement_df = disagreement_df.drop(['User', 'Hashtag', 'Date', 'Vader & Transformer Comparison', 'Vader & TextBlob Comparison',
 'Transformer & TextBlob Comparison', 'Total Disagreement'], axis = 1)

#disagreement_df.to_csv('outlierTweets.csv', mode='a', index=True)#, header=False)


