"""
Copyright 2021 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import argparse
import numpy as np
import os
import logging
import time
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.train import Checkpoint

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
                
                               
## Compile the model by setting the optimizer, loss function and metrics
def compile_model(model, learning_rate, decay):
    logger.info('Compiling the model...')
    # Instantiate the optimizer
    optimizer = Adam(learning_rate=learning_rate, decay=decay)
    # Instantiate the loss function
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the metrics
    train_acc_metric = SparseCategoricalAccuracy()
    val_acc_metric = SparseCategoricalAccuracy()
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[train_acc_metric])
    logger.info('Completed compiling the model.')
    return optimizer, loss_fn, train_acc_metric, val_acc_metric


## Prepare the batch datasets
def prepare_batch_datasets(x_train, y_train, batch_size):
    logger.info('Preparing train and validation datasets for batches...')
    # Reserve the required samples for validation
    x_val = x_train[-(len(x_train) * int(VALIDATION_DATA_SPLIT)):]
    y_val = y_train[-(len(y_train) * int(VALIDATION_DATA_SPLIT)):]
    # Prepare the training dataset with shuffling
    train_dataset = Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Prepare the validation dataset
    val_dataset = Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    logger.info('Completed preparing train and validation datasets for batches.')
    return x_val, y_val, train_dataset, val_dataset
    

## Define the training step
@tf.function
def training_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer
        logits = model(x_batch_train, training=True)
        # Compute the loss value
        loss_value = loss_fn(y_batch_train, logits)
    # Retrieve the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Update training metric
    train_acc_metric.update_state(y_batch_train, logits)
    return loss_value
    
    
## Define the validation step
@tf.function
def validation_step(model, x_batch_val, y_batch_val, val_acc_metric):
    val_logits = model(x_batch_val, training=False)
    val_acc_metric.update_state(y_batch_val, val_logits)
    
    
## Perform validation
def perform_validation(model, val_dataset, val_acc_metric):
    logger.debug('Performing validation...')
    for x_batch_val, y_batch_val in val_dataset:
        validation_step(model, x_batch_val, y_batch_val, val_acc_metric)
    logger.debug('Completed performing validation.')
    return val_acc_metric.result()


## Save the model as a checkpoint
def save_checkpoint(checkpoint):
    logger.debug('Saving model checkpoint...')
    checkpoint.save(os.path.join(args.checkpoint_local_dir, 'tf2-checkpoint'))
    logger.info('Checkpoint counter = {}'.format(checkpoint.save_counter.numpy()))
    logger.debug('Completed saving model checkpoint.')
                
                               
## Train the model
def train_model(model, model_dir, x_train, y_train, batch_size, epochs, learning_rate, decay):
    history = []
    
    # Compile the model
    optimizer, loss_fn, train_acc_metric, val_acc_metric = compile_model(model, learning_rate, decay)
    
    # Create the checkpoint object
    if args.checkpoint_enabled.lower() == 'true':
        checkpoint = Checkpoint(model)
    
    # Prepare the batch datasets
    x_val, y_val, train_dataset, val_dataset = prepare_batch_datasets(x_train, y_train, batch_size)
    
    # Perform training
    logger.info('Training the model...')
    training_start_time = time.time()
    logger.debug('Iterating over epochs...')
    # Iterate over epochs
    for epoch in range(epochs):
        logger.debug('Starting epoch {}...'.format(int(epoch) + 1))
        epoch_start_time = time.time()
        
        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            logger.debug('Running training step {}...'.format(int(step) + 1))
            loss_value = training_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric)
            logger.debug('Training loss in step = {}'.format(loss_value))
            logger.debug('Completed running training step {}.'.format(int(step) + 1)) 
        
        # Perform validation and save metrics at the end of each epoch
        history.append([int(epoch) + 1, train_acc_metric.result(),
                        perform_validation(model, val_dataset, val_acc_metric)])
        # Reset metrics
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        
        # Save the model as a checkpoint
        if args.checkpoint_enabled.lower() == 'true':
            save_checkpoint(checkpoint)
        
        epoch_end_time = time.time()
        logger.debug("Epoch duration = %.2f second(s)" % (epoch_end_time - epoch_start_time))
        logger.debug('Completed epoch {}.'.format(int(epoch) + 1))
        
    logger.debug('Completed iterating over epochs.')
    training_end_time = time.time()
    logger.info('Training duration = %.2f second(s)' % (training_end_time - training_start_time))
    print_training_result(history)
    logger.info('Completed training the model.')

    
## Print training result
def print_training_result(history):
    output_table_string_list = []
    output_table_string_list.append('\n')
    output_table_string_list.append("{:<10} {:<25} {:<25}".format('Epoch', 'Accuracy', 'Validation Accuracy'))
    output_table_string_list.append('\n')
    size = len(history)
    for index in range(size):
        record = history[index]
        output_table_string_list.append("{:<10} {:<25} {:<25}".format(record[0], record[1], record[2]))
        output_table_string_list.append('\n')
    output_table_string_list.append('\n')
    logger.info(''.join(output_table_string_list))
    
    
## Evaluate the model
def evaluate_model(model, x_test, y_test):
    logger.info('Evaluating the model...')
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
    # Parse command-line arguments
    args, _ = parse_args()
    # Load train and test data
    x_train, y_train = load_and_preprocess_data('training', args.train, 'x_train.npy', 'y_train.npy')
    x_test, y_test = load_and_preprocess_data('test', args.test, 'x_test.npy', 'y_test.npy')
    # Create, train and evaluate the model
    model = create_model()
    if args.checkpoint_load_previous.lower() == 'true':
        load_weights_from_latest_checkpoint(model)
    train_model(model, args.model_dir, x_train, y_train, args.batch_size, args.epochs, args.learning_rate, args.decay)
    evaluate_model(model, x_test, y_test)
    # Save the generated model
    save_model(model, args.model_dir)
    logger.info('Completed executing the main() function.')