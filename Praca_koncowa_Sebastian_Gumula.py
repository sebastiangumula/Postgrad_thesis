# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 07:58:50 2024

@author: MSI
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None

sales = pd.read_csv('retail_sales_dataset.csv')
clothing = pd.read_csv('women_clothing_ecommerce_sales.csv')
reviews = pd.read_csv('reviews.csv')

sales['Date'] = pd.to_datetime(sales['Date'])
clothing['order_date'] = pd.to_datetime(clothing['order_date'])
sales = sales.sort_values(by='Date')
clothing = clothing.sort_values(by='order_date')

print(sales)
print(clothing)
print(reviews)


"""
Jakie pytania ma biznes do mnie?

Biznes chce lepiej zrozumieć swoich klientów, prosi o analizę 

dla każdego datasetu

Zrób EDA:

    
Po EDA wyznacz:
    


Rozkład wieku w ujęciu kategorii

Sumę zysków z kategorii, płci itd 

KOBIETY ulubiony kolor, rozmiar, najbardziej dochodowe

ML

Zbadanie boxem jenkinsem prognozy salesów dla każdej kategorii (miesięczna) 

Prognoza arimką i lstmem

oszacowanie rmse mape itd i modlimy sie żeby działało

Analiza sentymentu z tego datasetu z komentarzami

"""

############################################################################################
#Dataset 1:

#EDA:
print(sales.info())    
print(sales.describe())

#Procentowy udział płci w każdej kategorii

#Sumę zysków z kategorii, płci itd 

#Wyodębnij time seriesa dla każdej kategorii i zrób miesięcznego

#Sprawdź box-jenkinsem

#prognoza

############################################################################################

#Dataset 2:
    
#EDA:
    
#KOBIETY ulubiony kolor, rozmiar, najbardziej dochodowe


#########################################################################################
#Dataset 3:
    
#EDA:
    
#Analiza sentymentu     



    
    
    
    
    
    