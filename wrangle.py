# '''Aquire and Prepare telco data from Codeup SQL database'''

import os
import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

from env import user, password, host

######################### Acquire main function #########################

def get_db_url(db):
    '''
    This function calls the username, password, and host from env file and provides database argument for SQL
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#-------------------------**Telco_Churn DATA** ```FROM SQL```-------------------------

def new_telco_churn_df():
    '''
    This function reads the telco_churn (NOT telco_normalized) data from the Codeup database into a DataFrame.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM customers LEFT JOIN internet_service_types USING (internet_service_type_id) LEFT JOIN contract_types USING (contract_type_id) LEFT JOIN payment_types USING (payment_type_id) LEFT JOIN customer_churn USING (customer_id) LEFT JOIN customer_contracts USING (customer_id) LEFT JOIN customer_details USING (customer_id) LEFT JOIN customer_payments USING (customer_id) LEFT JOIN customer_signups USING (customer_id) LEFT JOIN customer_subscriptions USING (customer_id)'
    
    # Read in DataFrame from Codeup db using defined functions.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df

def get_telco_churn_df():
    '''
    This function reads in telco_churn (NOT telco_normalized) data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''
    # Checks for csv file existence
    if os.path.isfile('telco_churn_df.csv'):
        
        # If csv file exists, reads in data from the csv file.
        df = pd.read_csv('telco_churn_df.csv', index_col=0)
        
    else:
        
        # If csv file does not exist, uses new_telco_churn_df function to read fresh data from telco db into a DataFrame
        df = new_telco_churn_df()
        
        # Cache data into a new csv file
        df.to_csv('telco_churn_df.csv')
        
    return df
