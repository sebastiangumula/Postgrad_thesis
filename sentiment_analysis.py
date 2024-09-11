# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 08:49:06 2024

@author: Sebastian Gumula
"""

from textblob import TextBlob
import pandas as pd

pd.options.display.max_columns = None
df = pd.read_csv("reviews.csv")

# Define a function for sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0:
        return "Positive"
    elif sentiment_polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to the 'Text' column and create a new 'Sentiment' column
df['Review Text'] = df['Review Text'].astype(str)
df['Sentiment'] = df['Review Text'].apply(analyze_sentiment)

# Print the DataFrame with sentiment analysis results
bins = [18, 22, 28, 33, 40, 45, 50, 55, 100]
bins2 =[1,2,3,4,5,6,7,8]
labels = ['Early Adult Transition', 
          'Entering the Adult World',
          'Age 30 Transition',
          'Settling Down',
          'Mid-Life Transition',
          'Entering the Middle Years',
          'Age 50 Transition',
          'Late Adulthood'
          ]


df['Levinson Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

result_df = df.groupby(['Levinson Age Group', 'Sentiment']).size().unstack(fill_value=0)
result_df.reset_index()
result_df.to_csv('%withingroup.csv')
