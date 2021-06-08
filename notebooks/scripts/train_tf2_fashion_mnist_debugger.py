"""
Copyright 2021 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import argparse
import numpy as np
import os
import logging
import smdebug
import smdebug.tensorflow as smd
import time
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Declare constants
TRAIN_VERBOSE_LEVEL = 0
EVALUATE_VERBOSE_LEVEL = 0
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
VALIDATION_DATA_SPLIT = 0.1

# Create the logger
logger = logging.getLogger(__name__)
logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))


## Parse and load the command-line arguments sent to the script
## These will be sent by SageMaker when it launches the training container
def parse_args():
    logger.info('Parsing command-line arguments...')
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--decay', type=float, default=1e-6)
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    # Model output directory
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # Checkpoint info
    parser.add_argument('--checkpoint_enabled', type=str, default='True')
    parser.add_argument('--checkpoint_load_previous', type=str, default='True')
    parser.add_argument('--checkpoint_local_dir', type=str, default='/opt/ml/checkpoints/')
    logger.info('Completed parsing command-line arguments.')
    return parser.parse_known_args()


## Initialize the SMDebugger for the Tensorflow framework
def init_smd():
    logger.info('Initializing the SMDebugger for the Tensorflow framework...')
    # Use KerasHook - the configuration file will be copied to /opt/ml/input/config/debughookconfig.json
    # automatically by SageMaker when it launches the training container
    hook = smd.KerasHook.create_from_json_file()
    logger.info('Debugger hook collections :: {}'.format(hook.get_collections()))
    logger.info('Completed initializing the SMDebugger for the Tensorflow framework.')
    return hook


## Load data from local directory to memory and preprocess
def load_and_preprocess_data(data_type, data_dir, x_data_file_name, y_data_file_name):
    logger.info('Loading and preprocessing {} data...'.format(data_type))
    x_data = np.load(os.path.join(data_dir, x_data_file_name))
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    y_data = np.load(os.path.join(data_dir, y_data_file_name))
    logger.info('Completed loading and preprocessing {} data.'.format(data_type))
    return x_data, y_data


## Construct the network
def create_model():
    logger.info('Creating the model...')
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', 
               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),        
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),    
        MaxPooling2D(pool_size=(2, 2)),   
        
        Flatten(),
        
        Dense(1024, activation='relu'),
        
        Dense(512, activation='relu'),
        
        Dense(NUM_CLASSES, activation='softmax')
    ])
    # Print the model summary
    logger.info(model.summary())
    logger.info('Completed creating the model.')
    return model
    

## Load the weights from the latest checkpoint
def load_weights_from_latest_checkpoint(model):
    file_list = os.listdir(args.checkpoint_local_dir)
    logger.info('Checking for checkpoint files...')
    if len(file_list) > 0:
        logger.info('Checkpoint files found.')
        logger.info('Loading the weights from the latest model checkpoint...')
        model.load_weights(tf.train.latest_checkpoint(args.checkpoint_local_dir))
        logger.info('Completed loading weights from the latest model checkpoint.')
    else:
         logger.info('Checkpoint files not found.')
                
                               
## Compile the model by setting the loss and optimizer functions
def compile_model(model, learning_rate, decay):
    logger.info('Compiling the model...')
    optimizer = Adam(learning_rate=learning_rate, decay=decay)
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    logger.info('Completed compiling the model.')

    
## Train the model
def train_model(model, model_dir, x_train, y_train, batch_size, epochs):
    logger.info('Training the model...')
    hook.set_mode(smd.modes.TRAIN)
    # SMDebugger: Save basic details
    hook.save_scalar('batch_size', batch_size, sm_metric=True)
    hook.save_scalar('number_of_epochs', epochs, sm_metric=True)
    hook.save_scalar('train_steps_per_epoch', len(x_train) / batch_size)
    # Check for checkpointing and process accordingly
    if args.checkpoint_enabled.lower() == 'true':
        logger.info('Initializing to perform checkpointing...')
        checkpoint = ModelCheckpoint(filepath=os.path.join(args.checkpoint_local_dir, 'tf2-checkpoint-{epoch}'),
                                     save_best_only=False, save_weights_only=True,
                                     save_frequency='epoch',
                                     verbose=TRAIN_VERBOSE_LEVEL)
        callbacks = [checkpoint, hook]
        logger.info('Completed initializing to perform checkpointing.')
    else:
        logger.info('Checkpointing will not be performed.')
        callbacks = [hook]
    training_start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=VALIDATION_DATA_SPLIT, validation_freq=1,
                        callbacks=callbacks, verbose=TRAIN_VERBOSE_LEVEL)
    training_end_time = time.time()
    logger.info('Training duration = %.2f second(s)' % (training_end_time - training_start_time))
    print_training_result(history.history)
    logger.info('Completed training the model.')


## Print training result
def print_training_result(history):
    loss = history["loss"]
    accuracy = history["accuracy"]
    val_loss = history["val_loss"]
    val_accuracy = history["val_accuracy"]
    size = len(accuracy)
    output_table_string_list = []
    output_table_string_list.append('\n')
    output_table_string_list.append("{:<10} {:<25} {:<25} {:<25} {:<25}".format('Epoch', 'Training Loss',
                                                                                'Training Accuracy',
                                                                                'Validation Loss',
                                                                                'Validation Accuracy'))
    output_table_string_list.append('\n')
    for index in range(size):
        output_table_string_list.append("{:<10} {:<25} {:<25} {:<25} {:<25}".format(index + 1,
                                                                                   loss[index],
                                                                                   accuracy[index],
                                                                                   val_loss[index],
                                                                                   val_accuracy[index]))
        output_table_string_list.append('\n')
    output_table_string_list.append('\n')
    logger.info(''.join(output_table_string_list))

    
## Evaluate the model
def evaluate_model(model, x_test, y_test):
    logger.info('Evaluating the model...')
    hook.set_mode(smd.modes.EVAL)
    test_loss, test_accuracy = model.evaluate(x_test, y_test,
                                              verbose=EVALUATE_VERBOSE_LEVEL)
    logger.info('Test loss = {}'.format(test_loss))
    logger.info('Test accuracy = {}'.format(test_accuracy))
    logger.info('Completed evaluating the model.')
    return test_loss, test_accuracy

    
## Save the model
def save_model(model, model_dir):
    logger.info('Saving the model...')
    tf.saved_model.save(model, model_dir)
    logger.info('Completed saving the model.')

    
## The main function
if __name__ == "__main__":
    logger.info('Executing the main() function...')
    logger.info('TensorFlow version : {}'.format(tf.__version__))
    logger.info('SMDebug version : {}'.format(smdebug.__version__))
    # Parse command-line arguments
    args, _ = parse_args()
    # Initialize the SMDebugger for the Tensorflow framework
    hook = init_smd()
    # Load train and test data
    x_train, y_train = load_and_preprocess_data('training', args.train, 'x_train.npy', 'y_train.npy')
    x_test, y_test = load_and_preprocess_data('test', args.test, 'x_test.npy', 'y_test.npy')
    # Create, compile, train and evaluate the model
    model = create_model()
    if args.checkpoint_load_previous.lower() == 'true':
        load_weights_from_latest_checkpoint(model)
    compile_model(model, args.learning_rate, args.decay)
    train_model(model, args.model_dir, x_train, y_train, args.batch_size, args.epochs)
    evaluate_model(model, x_test, y_test)
    # Save the generated model
    save_model(model, args.model_dir)
    # Close the SMDebugger hook
    hook.close()
    logger.info('Completed executing the main() function.')