#!/usr/bin/env python
# coding: utf-8

# ######################### EXPLORE #########################

# In[15]:


# IMPORTS NEEDED FOR EXPLORATION
import pandas as pd
import numpy as np


# In[17]:


def explore_toc ():
    """
    PRINT TABLE OF CONTENTS FOR CUSTOM EXPLORE FUNCTIONS
    """
    print("** CUSTOM EXPLORATION FUNCTIONS")
    
    print("explore_tips: PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EXPLORATION")
    print("nunique_column_all(df): PRINT NUNIQUE OF ALL COLUMNS")
    print("nunique_column_objects(df): PRINT NUNIQUE OF COLUMNS THAT ARE OBJECTS")
    print("nunique_column_qty(df): PRINT NUNIQUE OF COLUMNS THAT ARE *NOT* OBJECTS")
    print("numeric_range(df): COMPUTE RANGE FOR ALL NUMERIC VARIABLES")


# In[16]:


def explore_tips():

    """
    PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EXPLORATION

    """
    print("** USEFUL EXPLORATORY CODE**")
    print ("DFNAME.head()")
    print ("DFNAME.shape")
    print ("DFNAME.shape[0] #read row count")
    print ("DFNAME.describe().T")
    print ("DFNAME.columns.to_list()")
    print("DFNAME.COLUMNNAME.value_counts(dropna=False)")
    print ("DFNAME.dtypes")
    print("DFNAME.select_dtypes(include='object').columns")
    print("DFNAME.select_dtypes(include='float').columns")
    print("pd.crosstab(DFNAME.COLUMN-1, DFNAME.COLUMN-2)")


# In[ ]:


# PRINT NUNIQUE OF ALL COLUMNS

def nunique_column_all(df):
    for col in df.columns:
        print(df[col].value_counts())
        print()


# In[4]:


# PRINT NUNIQUE OF COLUMNS THAT ARE OBJECTS

def nunique_column_objects(df): 
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(f'{col} has {df[col].nunique()} unique values.')


# In[5]:


# PRINT NUNIQUE OF COLUMNS THAT ARE *NOT* OBJECTS

def nunique_column_qty(df): 
    for col in df.columns:
        if df[col].dtypes != 'object':
            print(f'{col} has {df[col].nunique()} unique values.')


# In[ ]:


# **COMPUTE RANGE FOR ALL NUMERIC VARIABLES**

def numeric_range(df):
    numeric_list = df.select_dtypes(include = 'float').columns.tolist()
    numeric_range = df[numeric_list].describe().T
    numeric_range['range'] = numeric_range['max'] - numeric_range['min']
    return numeric_range


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS FOR ALL "FLOAT" COLUMNS

# float_cols = train_iris.select_dtypes(include='float').columns

# Plot numeric columns
#plot_float_cols = float_cols 
#for col in plot_float_cols:
#    plt.hist(train_iris[col])
#    plt.title(col)
#    plt.show()
#    plt.boxplot(train_iris[col])
#    plt.title(col)
#    plt.show()


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS FOR ALL "OBJECT" COLUMNS

# train.species.value_counts()
# plt.hist(train_iris.species_name)


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS

#test_var = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#for var in test_var:
#    t_stat, p_val = t_stat, p_val = stats.mannwhitneyu(virginica[var], versicolor[var], alternative="two-sided")
#    print(f'Comparing {var} between Virginica and Versicolor')
#    print(t_stat, p_val)
#    print('')
#    print('---------------------------------------------------------------------')
#    print('')


# In[ ]:


# sns.pairplot(DF, hue='TARGET_COLUMN', corner=True)
# plt.show()


# In[ ]:


# BUILD A FUNCTION; This will list out Accuracies for each model

# accuracy_dictionary = {'Baseline': (petpics_df.actual == petpics_df.baseline).mean(), 
#                   'Model_1 accuracy': (petpics_df.actual == petpics_df.model1).mean(),
#                   'Model_2 accuracy': (petpics_df.actual == petpics_df.model2).mean(),
#                   'Model_3 accuracy': (petpics_df.actual == petpics_df.model3).mean(),
#                   'Model_4 accuracy': (petpics_df.actual == petpics_df.model4).mean()}
# accuracy_dictionary


# ```
# {'Baseline': 0.6508,
#  'Model_1 accuracy': 0.8074,
#  'Model_2 accuracy': 0.6304,
#  'Model_3 accuracy': 0.5096,
#  'Model_4 accuracy': 0.7426}
#  ```

# ######################### EVALUATE #########################

# In[11]:


# IMPORTS NEEDED FOR EVALUATION
from sklearn.metrics import confusion_matrix


# In[28]:


def eval_toc ():
    """
    PRINT TABLE OF CONTENTS FOR CUSTOM EVALUATION FUNCTIONS
    """
    print("** CUSTOM EVALUATION FUNCTIONS")
    
    print("eval_tips: PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EXPLORATION")
    print("print_classification_metrics(actuals, predictions): PRINT CLASSIFICATION METRICS FROM CONFUSION MATRIX")


# In[29]:


def eval_tips():
    """
    PRINT A LIST OF USEFUL FUNCTIONS, METHODS, AND ATTRIBUTES USED FOR EXPLORATION
    """
    print("** USEFUL EXPLORATORY CODE**")

#    print ("DFNAME.head()")
#    print ("DFNAME.shape")
#    print ("DFNAME.shape[0] #read row count")
#    print ("DFNAME.describe().T")
#    print ("DFNAME.columns.to_list()")
#    print("DFNAME.COLUMNNAME.value_counts(dropna=False)")
#    print ("DFNAME.dtypes")
#    print("DFNAME.select_dtypes(include='object').columns")
#    print("DFNAME.select_dtypes(include='float').columns")
#    print("pd.crosstab(DFNAME.COLUMN-1, DFNAME.COLUMN-2)")


# In[12]:


def print_classification_metrics(actuals, predictions):
    """
    Stolen from Codeup Instructor Ryan McCall (Thanx, this is awesome!)\n
    This function extrapulates the Confusion Matrix metrics (TN, FP, FN, TP) using the method .ravel from sklearn\n
    then simply applies those metrics to the appropriate formulas to return Accuracy, Precision, Recall and Fl.
    """
   
    TN, FP, FN, TP = confusion_matrix(actuals, predictions).ravel()
    ALL = TP + TN + FP + FN
    
    accuracy = (TP + TN)/ALL
    print(f"Accuracy: {accuracy}")

    true_positive_rate = TP/(TP+FN)
    print(f"True Positive Rate: {true_positive_rate}")

    false_positive_rate = FP/(FP+TN)
    print(f"False Positive Rate: {false_positive_rate}")

    true_negative_rate = TN/(TN+FP)
    print(f"True Negative Rate: {true_negative_rate}")

    false_negative_rate = FN/(FN+TP)
    print(f"False Negative Rate: {false_negative_rate}")

    precision = TP/(TP+FP)
    print(f"Precision: {precision}")

    recall = TP/(TP+FN)
    print(f"Recall: {recall}")

    f1_score = 2*(precision*recall)/(precision+recall)
    print(f"F1 Score: {f1_score}")

    support_pos = TP + FN
    print(f"Support (0): {support_pos}")

    support_neg = FP + TN
    print(f"Support (1): {support_neg}")


# In[ ]:


# Wraps Codeup Instructor Ryan McCall's Classification Matrix Function into a single print event with Named Headers.

# Actual = 


#print('** Baseline:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.baseline))

#print('')
#print('** Model_1:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model1))

#print('')
#print('** Model_2:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model2))

#print('')
#print('** Model_3:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model3))

#print('')
#print('** Model_4:')
#print(print_classification_metrics(petpics_df.actual, petpics_df.model4))

