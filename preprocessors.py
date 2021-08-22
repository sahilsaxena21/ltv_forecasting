import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoder = LabelEncoder()):
        self.encoder = encoder

    def fit(self, X, y=None):
        # note that x is the target in this case
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = np_utils.to_categorical(self.encoder.transform(X))
        return X

def _encoded_features(x):
  return int((x/20) + 1)

def _sequence_vectorization(df):
    # note that x_train, y_train are already sorted by customer_labels

    #reset index and deduplicate entries
    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace=True)

    #for each transaction, calculate days since first transaction for the given customer, time_t
    df["first_date"] = df.groupby("customer_labels")["time"].transform("min")
    df["days_since_first_transaction"] = df["time"] - df["first_date"]
    df["days_since_first_transaction"] = df["days_since_first_transaction"].apply(lambda x: int(str(x).split()[0]))
    df["days_since_first_transaction"] = df["days_since_first_transaction"].apply(lambda x: _encoded_features(x))

    df.drop(["time", "first_date"], inplace = True, axis = 1)

    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace = True)

    pivoted = df.pivot(index = "customer_labels", 
                        columns = "days_since_first_transaction", 
                        values = "days_since_first_transaction")
    pivoted = pivoted.reset_index()

    #ignore the customer_labels column
    cols_list = pivoted.columns.tolist()[1:]

    #start sequence for all transactions is 1, representing day 1
    pivoted[cols_list[0]] = 1
    pivoted.fillna(0, inplace = True)
    pivoted['sequence'] = pivoted[cols_list].apply(lambda row: row.values.tolist(), axis=1)
    final_df = pivoted[["sequence", "customer_labels"]]

    with pd.option_context('mode.chained_assignment', None):

        final_df["sequence"] = final_df["sequence"].apply(lambda x: [i for i in x if i != 0])
        final_df["sequence"] = final_df["sequence"].apply(lambda x: [int(i) for i in x])
        final_df["sequence"] = final_df["sequence"].apply(lambda x: np.array(x))

    #initiate sequence_df
    sequence_df = pd.DataFrame()
    sequence_df = final_df[["sequence", "customer_labels"]].copy()
    sequence_df.reset_index(drop = True, inplace = True)

    #make a deduplicated dataframe from the original X dataframe (i.e. before the pivot operation)
    df_target_deduplicated = df.drop_duplicates(subset = "customer_labels")
    df_target_deduplicated.sort_values(by = "customer_labels", ascending = True, inplace = True)
    df_target_deduplicated.reset_index(drop = True,inplace = True)

    if "customer_label_count" in df_target_deduplicated.columns:
      #means if this is the training set containing the oversampling column
      df_target_deduplicated = df_target_deduplicated[["customer_labels", "customer_label_count"]]
    
    else:
      #if this is the test set which does not contain an oversampling column
      df_target_deduplicated = df_target_deduplicated[["customer_labels"]]

    #merge the customer_label_count column
    sequence_df = pd.merge(sequence_df, df_target_deduplicated, on=['customer_labels'], how='inner')
    sequence_df.sort_values(by = "customer_labels", ascending = True, inplace=True)
    sequence_df.reset_index(drop = True, inplace = True)
    print(sequence_df.shape)
    return sequence_df

class CreateDataset(BaseEstimator, TransformerMixin):

    def __init__(self, customer_labels_var, time_var, maxlen, padding):
        self.maxlen = maxlen
        self.padding = padding

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        #sequence_vectors is of type array of form [[1], [1,5]]
        sequence_df = _sequence_vectorization(X)

        # print(repeat_count_list[1692])
        #pad the sequence values
        sequence_vectors = np.array(sequence_df["sequence"])
        padded_sequence = pad_sequences(sequence_vectors, truncating='pre', maxlen = self.maxlen, padding = self.padding)
        
        if "customer_label_count" in sequence_df.columns:
          #here we repeat rows based on oversampling
          #store repeat count in list for train set
          repeat_count_list = sequence_df["customer_label_count"].tolist()
          X = np.repeat(padded_sequence, repeat_count_list, axis=0)
        else:
          X = padded_sequence

        print(X.shape)

        print('Sequence shape: {} size: {:,}'.format(X.shape, X.size))

        #transform to suit the Keras model with n_features = 1
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        return X


# def _encoded_features(x):
#   return int((x/20) + 1)

# def _sequence_vectorization(df):
#     # note that x_train, y_train are already sorted by customer_labels

#     #reset index and deduplicate entries
#     df.drop_duplicates(inplace = True)
#     df.reset_index(drop = True, inplace=True)

#     #for each transaction, calculate days since first transaction for the given customer, time_t
#     df["first_date"] = df.groupby("customer_labels")["time"].transform("min")
#     df["days_since_first_transaction"] = df["time"] - df["first_date"]
#     df["days_since_first_transaction"] = df["days_since_first_transaction"].apply(lambda x: int(str(x).split()[0]))
#     df["days_since_first_transaction"] = df["days_since_first_transaction"].apply(lambda x: _encoded_features(x))

#     df.drop(["time", "first_date"], inplace = True, axis = 1)

#     df.drop_duplicates(inplace = True)
#     df.reset_index(drop = True, inplace = True)

#     pivoted = df.pivot(index = "customer_labels", 
#                         columns = "days_since_first_transaction", 
#                         values = "days_since_first_transaction")
#     pivoted = pivoted.reset_index()

#     #ignore the customer_labels column
#     cols_list = pivoted.columns.tolist()[1:]

#     #start sequence for all transactions is 1, representing day 1
#     pivoted[cols_list[0]] = 1
#     pivoted.fillna(0, inplace = True)
#     pivoted['sequence'] = pivoted[cols_list].apply(lambda row: row.values.tolist(), axis=1)
#     final_df = pivoted[["sequence", "customer_labels"]]

#     with pd.option_context('mode.chained_assignment', None):

#         final_df["sequence"] = final_df["sequence"].apply(lambda x: [i for i in x if i != 0])
#         final_df["sequence"] = final_df["sequence"].apply(lambda x: [int(i) for i in x])
#         final_df["sequence"] = final_df["sequence"].apply(lambda x: np.array(x))

#     #make a deduplicated dataframe from the original X dataframe (i.e. before the pivot operation)
#     df_target_deduplicated = df.drop_duplicates(subset = "customer_labels")
#     df_target_deduplicated.sort_values(by = "customer_labels", ascending = True, inplace = True)
#     df_target_deduplicated.reset_index(drop = True,inplace = True)
#     df_target_deduplicated = df_target_deduplicated[["customer_labels", "customer_label_count"]]

#     #initiate sequence_df
#     sequence_df = pd.DataFrame()
#     sequence_df = final_df[["sequence", "customer_labels"]].copy()
#     sequence_df.reset_index(drop = True, inplace = True)

#     #merge the customer_label_count column
#     sequence_df = pd.merge(sequence_df, df_target_deduplicated, on=['customer_labels'], how='inner')
#     sequence_df.sort_values(by = "customer_labels", ascending = True, inplace=True)
#     sequence_df.reset_index(drop = True, inplace = True)
#     print(sequence_df.shape)
#     return sequence_df

# class CreateDataset(BaseEstimator, TransformerMixin):

#     def __init__(self, customer_labels_var, time_var, maxlen, padding):
#         self.maxlen = maxlen
#         self.padding = padding

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         #sequence_vectors is of type array of form [[1], [1,5]]
#         sequence_df = _sequence_vectorization(X)

#         #store repeat count in list
#         repeat_count_list = sequence_df["customer_label_count"].tolist()
#         # print(repeat_count_list[1692])

#         #pad the sequence values
#         sequence_vectors = np.array(sequence_df["sequence"])
#         padded_sequence = pad_sequences(sequence_vectors, truncating='pre', maxlen = self.maxlen, padding = self.padding)

#         #here we repeat rows based on oversampling
#         X = np.repeat(padded_sequence, repeat_count_list, axis=0)
#         print(X.shape)

#         print('Sequence shape: {} size: {:,}'.format(X.shape, X.size))

#         #transform to suit the Keras model with n_features = 1
#         n_features = 1
#         X = X.reshape((X.shape[0], X.shape[1], n_features))

#         return X

class RandomOverSamplerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self):
        return self

    def transform(self):

      df_deduplicated = self.X.drop_duplicates(subset = "customer_labels", keep = 'first')
      df_deduplicated.sort_values(by = "customer_labels", ascending = True, inplace = True)
      df_deduplicated.reset_index(drop = True, inplace = True)
      self.y.reset_index(drop = True, inplace = True)

      #at this point, X_deduplicated.shape[0] = y.shape[0]
      #xtrain_deplicated and y should match 1-1
      df_deduplicated["target_encoded"] = self.y.copy()

      #oversample
      oversample = RandomOverSampler(random_state=43)
      X_temp, y_temp = oversample.fit_resample(df_deduplicated[["customer_labels","target_encoded"]], self.y)


      df_temp = pd.DataFrame(X_temp, columns=['customer_labels', 
                            'target_encoded'])

      #drop target from x
      df_temp.drop(["target_encoded"], inplace = True, axis = 1)

      #add count column
      df_temp["customer_label_count"] = df_temp.groupby(df_temp["customer_labels"])["customer_labels"].transform("count")

      #remove duplicates, because now we have count to store the number of repetitions. Also helps in join to ensure X has more rows than df_temp
      df_temp.drop_duplicates(subset = "customer_labels", inplace = True, keep = "first")

      #join
      X = pd.merge(self.X, df_temp, on = "customer_labels", how = "inner")

      return X, y_temp

if __name__ == '__main__':
    
    import data_management as dm
    import pandas as pd
    import config

#dm preprocessing

    #purchases
    purchases_df = dm.load_dataset(file_name=config.DATA_FILE)

    #binning the target col to represent 2 (high), 1 (medium), 0 (low transactions)
    purchases_df = dm.bin_transactions(purchases_df) 

    #train_test split
    customer_labels_train, customer_labels_test = dm.get_train_test_customer_labels(purchases_df)
    X_train, X_test = dm.X_subset_records(purchases_df, customer_labels_train, customer_labels_test)
    y_train, y_test = dm.y_subset_records(purchases_df, customer_labels_train, customer_labels_test)
    

#random oversampler
    #X_train: makes a new col with counts for each customer_label to balance dataset
    #y_train is oversampled

    enc = RandomOverSamplerTransformer(X_train, y_train)
    X_train, y_train = enc.transform()

#y_train to [1,0], [0,1] for the model fitting process only
    enc = TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    
    dataCreator = CreateDataset(customer_labels_var = config.CUSTOMER_LABEL_VAR, 
                                time_var = config.TIME_VAR, maxlen = config.MAX_LEN, 
                                padding = config.PADDING)

    X_train = dataCreator.transform(X_train)    
    print(X_train.shape)

