#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from glob import glob
import pathlib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# import sklearn.external.joblib as extjoblib
import joblib
# from sklearn.externals import joblib

from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier

import model as m
import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    path = r'C:\Users\Sahil\Documents\cnn\my_project\dataset'
    file_path = os.path.join(path,file_name)
    dataframe = pd.read_csv(file_path, index_col=0, parse_dates=[config.TIME_VAR])
    dataframe.reset_index(drop = True, inplace = True)
    dataframe = dataframe.dropna(how = "any", thresh = None)
    dataframe.drop_duplicates(inplace = True)

    return dataframe

def _encode_transactions(x):
    if x <= config.LOW_SALES:
      return 0
    elif x > config.LOW_SALES:
      return 1

def bin_transactions(df):
  df[config.TARGET_VAR] = df[config.TRANSACTION_VAR].apply(lambda x: _encode_transactions(x))
  return df

def get_train_test_customer_labels(df):
  df_split = df[[config.CUSTOMER_LABEL_VAR, config.TARGET_VAR]].copy()
  df_split.drop_duplicates(subset = [config.CUSTOMER_LABEL_VAR], inplace = True)
  train, test = train_test_split(df_split, test_size = 0.30, random_state = 42, stratify = df_split[config.TARGET_VAR])
  customer_labels_train = train[config.CUSTOMER_LABEL_VAR].tolist()
  customer_labels_test = test[config.CUSTOMER_LABEL_VAR].tolist()
  return customer_labels_train, customer_labels_test

def X_subset_records(df, customer_labels_train, customer_labels_test):
  #keep the duplicate entries for sequencing for x_train and x_test
  df_train_seq = df[df[config.CUSTOMER_LABEL_VAR].isin(customer_labels_train)]
  df_test_seq = df[df[config.CUSTOMER_LABEL_VAR].isin(customer_labels_test)]

  #these are sorted by customer_labels to ensure X,y are 1 to 1
  df_train_seq.sort_values(by = config.CUSTOMER_LABEL_VAR, ascending=True, inplace=True)
  df_test_seq.sort_values(by = config.CUSTOMER_LABEL_VAR, ascending=True, inplace=True)

  X_train, X_test = df_train_seq[[config.CUSTOMER_LABEL_VAR, config.TIME_VAR]], df_test_seq[[config.CUSTOMER_LABEL_VAR, config.TIME_VAR]]

  X_train.reset_index(drop=True, inplace=True)
  X_test.reset_index(drop=True, inplace=True)

  return X_train, X_test


def y_subset_records(df, customer_labels_train, customer_labels_test):
  #without duplicates for the y_train, y_test

  #use df_split because it does not have duplicates
  df_split = df[[config.CUSTOMER_LABEL_VAR, config.TARGET_VAR]].copy()
  df_split.drop_duplicates(subset = [config.CUSTOMER_LABEL_VAR], inplace = True)

  df_train = df_split[df_split[config.CUSTOMER_LABEL_VAR].isin(customer_labels_train)]
  df_test = df_split[df_split[config.CUSTOMER_LABEL_VAR].isin(customer_labels_test)]

  #these are sorted by customer_labels to ensure X,y are 1 to 1
  df_train.sort_values(by = config.CUSTOMER_LABEL_VAR, ascending=True, inplace=True)
  df_test.sort_values(by = config.CUSTOMER_LABEL_VAR, ascending=True, inplace=True)

  y_train, y_test = df_train[config.TARGET_VAR], df_test[config.TARGET_VAR]

  y_train.reset_index(drop=True, inplace=True)
  y_test.reset_index(drop=True, inplace=True)

  return y_train, y_test


def save_pipeline_keras(model):
    
    joblib.dump(model.named_steps['dataset'], config.PIPELINE_PATH)
    joblib.dump(model.named_steps['lstm_model'].classes_, config.CLASSES_PATH)
    model.named_steps['lstm_model'].model.save(config.MODEL_PATH)
    
    
def load_pipeline_keras():
    dataset = joblib.load(config.PIPELINE_PATH)
    
    build_model = lambda: load_model(config.MODEL_PATH)
    
    classifier = KerasClassifier(build_fn=build_model,
                          validation_split=0.2,
                          epochs=config.EPOCHS,
                          verbose=2,
                          callbacks=m.callbacks_list
                          )
    
    classifier.classes_ = joblib.load(config.CLASSES_PATH)
    classifier.model = build_model()
    
    return Pipeline([
        ('dataset', dataset),
        ('lstm_model', classifier)
    ])
    
    
if __name__ == '__main__':
    
    purchases_df = load_dataset(file_name=config.DATA_FILE)

    #binning the target col to represent 2 (high), 1 (medium), 0 (low transactions)
    purchases_df = bin_transactions(purchases_df) 

    #train_test split
    customer_labels_train, customer_labels_test = get_train_test_customer_labels(purchases_df)
    X_train, X_test = X_subset_records(purchases_df, customer_labels_train, customer_labels_test)
    y_train, y_test = y_subset_records(purchases_df, customer_labels_train, customer_labels_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    


# In[ ]:




