import data_management as dm
import config


def make_prediction(*, path_to_transactions) -> float:
    """Make a prediction using the saved model pipeline."""
    
    # Load data
    # create a dataframe with columns = ['time', 'customer_labels']
    # column "image" contains path to image 
    # columns target can contain all zeros, it doesn't matter
    
    #dataframe represents X_test
    dataframe = path_to_transactions # needs to load as above described
    pipe = dm.load_pipeline_keras()
    predictions = pipe.pipe.predict(dataframe)
    #response = {'predictions': predictions, 'version': _version}

    return predictions


if __name__ == '__main__':
    
    import joblib
    
    purchases_df = dm.load_dataset(file_name=config.DATA_FILE)

    #binning the target col to represent 2 (high), 1 (medium), 0 (low transactions)
    #this step is only needed to be able to carry out the rest of the steps
    purchases_df = dm.bin_transactions(purchases_df) 

    #get the X_test
    customer_labels_train, customer_labels_test = dm.get_train_test_customer_labels(purchases_df)
    X_train, X_test = dm.X_subset_records(purchases_df, customer_labels_train, customer_labels_test)
    
    # pipe = joblib.load(config.PIPELINE_PATH)
    pipe = dm.load_pipeline_keras()
    
    predictions = pipe.predict(X_test)
    print(predictions)