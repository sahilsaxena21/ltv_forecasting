from sklearn.pipeline import Pipeline

import config
import preprocessors as pp
import model

pipe = Pipeline([
                ('dataset', pp.CreateDataset(customer_labels_var = config.CUSTOMER_LABEL_VAR, 
                                            time_var = config.TIME_VAR, maxlen = config.MAX_LEN, 
                                            padding = config.PADDING)),
                ('lstm_model', model.lstm_clf)
                ])


if __name__ == '__main__':
    
    from sklearn.metrics import f1_score
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

    enc = pp.RandomOverSamplerTransformer(X_train, y_train)
    X_train, y_train = enc.transform()

#y_train to [1,0], [0,1] for the model fitting process only
    enc = pp.TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    
    pipe.fit(X_train, y_train)
    
    test_y = enc.transform(y_test)
    predictions = pipe.predict(X_test)
    
    acc = f1_score(enc.encoder.transform(y_test),
                   predictions,
                   average = "macro")
    
    print('F1-Score: ', acc)