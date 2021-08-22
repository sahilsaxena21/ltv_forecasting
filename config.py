# DATA
DATASET_DIR = 'dataset'
DATA_FILE = 'purchases.csv'
TIME_VAR = 'time'
CUSTOMER_LABEL_VAR = 'customer_labels'
TRANSACTION_VAR = "transactions"
TARGET_VAR = "binned_transactions"


#transactions equal to or less than this value is classified as "low" sales volume for the customer
LOW_SALES= 1



# MODEL FITTING
# IMAGE_SIZE = 150 # 50 for testing, 150 for final model
BATCH_SIZE = 10
PADDING = "post"
MAX_LEN = 10
EPOCHS = 10 # 1 for testing, 10 for final model

# MODEL PERSISTING
MODEL_PATH = "lstm_model.h5"
PIPELINE_PATH = 'lstm_pipe.pkl'
CLASSES_PATH = 'classes.pkl'
ENCODER_PATH = 'encoder.pkl'


