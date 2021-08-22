# for the convolutional network
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam 
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier


import config

def lstm_model(n_steps = 10, n_features = 1):
    model = Sequential([
        LSTM(64,activation='relu', input_shape=(n_steps, n_features)),
        Dense(2, activation='softmax')])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


checkpoint = ModelCheckpoint(config.MODEL_PATH,
                             monitor='val_accuracy',
                             verbose=0,
                             save_best_only=True,
                             mode='max')
                              
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              factor=0.5,
                              patience=2,
                              verbose=1,
                              mode='max',
                              min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]


lstm_clf = KerasClassifier(build_fn=lstm_model,
                          validation_split=0.2,
                          epochs=config.EPOCHS,
                          shuffle = False,
                          callbacks=callbacks_list,
                          verbose=0
                          )

                            
                              
if __name__ == '__main__':
    
    model = lstm_model(n_steps = 10, n_features = 1)
    model.summary()
    
#    import data_management as dm
#    import config
#    import preprocessors as pp
#    
#    model = lstm_model(image_size = config.IMAGE_SIZE)
#    model.summary()
#    
    # purchases_df = dm.load_dataset(file_name=config.DATA_FILE)
    # X_train, X_test, y_train, y_test = dm.get_train_test_target(purchases_df)
#    
#    enc = pp.TargetEncoder()
#    enc.fit(y_train)
#    y_train = enc.transform(y_train)
#    
#    dataset = pp.CreateDataset(50)
#    X_train = dataset.transform(X_train)
#    
#    lstm_clf.fit(X_train, y_train)