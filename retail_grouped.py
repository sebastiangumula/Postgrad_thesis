# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:41:56 2024

@author: Sebastian Gumula
"""
import pandas as pd

df = pd.read_csv("retail_sales_dataset.csv")

bins = [18, 22, 28, 33, 40, 45, 50, 55, 100]
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

age_groups = df.groupby('Levinson Age Group')['Age'].mean()

print(df)

df.to_csv('retail_grouped.csv')