import numpy as np
from keras.models import Input, Model
from keras.layers import Dense, Embedding, Flatten
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from random import choices
from keras.callbacks import EarlyStopping
import argparse
import sys
from numpy.random import seed
from tensorflow import set_random_seed
import keras.backend as K
            

def train(weights, bias, lr = 0.001, l1_reg = 0, l2_reg = 0.005, batch_size = 32, 
          epochs = 100, till_convergence = False):
    '''
        Function that uses activation maximisation to extract
        optimal embeddings for a set of words.
    '''
    
    
    print('L2_REG:', l2_reg)
    # Build  model
    opt = Adam(lr = float(lr))
    
    X_input = Input((1,), name = 'Input')
    X_tensor = Embedding(weights.shape[0], weights.shape[1], input_length = 1, #weights=[weights],
                         trainable = True, embeddings_regularizer = regularizers.L1L2(float(l1_reg), float(l2_reg)))(X_input)
    X_tensor = Flatten(name = 'flatten')(X_tensor)
    
    
    X_output = Dense(weights.shape[0], activation = 'softmax', name = 'softmax_out', trainable = False)(X_tensor)

    model = Model(inputs = X_input, outputs = X_output)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    print(model.get_weights()[0].shape)

    print(model.summary())
    # set softmax weights
    model.layers[-1].set_weights([weights.T, bias.T])

    # Train Model
    print('Training Model')

    X = list(range(weights.shape[0]))
    y = list(range(weights.shape[0]))
    if till_convergence == True:
        current_loss = 1000
        previous_loss = 10000
        while True:
            if abs(current_loss - previous_loss) < 0.005:
                    break
            model.fit(X, to_categorical(y, weights.shape[0]),
                epochs = 1,
                batch_size = batch_size,
                shuffle = True,
                verbose = 0)
            previous_loss = current_loss
            current_loss = model.history.history['loss'][-1]
            sys.stdout.write('\r' + ' Acc: ' + 
                             str(round(model.history.history['acc'][-1], 4)) 
                             + '    Loss: ' + str(round(model.history.history['loss'][-1], 4))
                             + '    Diff: ' + str(round(abs(current_loss - previous_loss), 4)))
                             
            if str(round(model.history.history['acc'][-1], 4)) == 1:
                break
            
        
    else:
        model.fit(X, to_categorical(y, weights.shape[0]),
            epochs = epochs,
            batch_size = batch_size,
            shuffle = True,
            verbose = 1)
    print('')

    return model.get_weights()[0]
    

if __name__ == '__main__':
    # Set up a few command line arguments for program
    parser = argparse.ArgumentParser(description = 'Module used for performing activation maximization.')

    parser.add_argument('--train', '-tn', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to build semantic model using activations maximization.'))

    parser.add_argument('--weight_path', '-wp', action = 'store',
                        required = False,
                        default = 'outputs',
                        help = ('Defines where to find dictionary with words and weights (must be a numpy dictionary).'))
    
    parser.add_argument('--bias_path', '-bp', action = 'store',
                        required = False,
                        default = 'outputs',
                        help = ('Defines where to find dictionary with words and bias (must be a numpy dictionary).'))

    parser.add_argument('--learning_rate', '-lr', action = 'store',
                        required = False,
                        default = '0.005',
                        help = ('Defines what learning rate to use.'))

    parser.add_argument('--l1_reg', '-l1', action = 'store',
                        required = False,
                        default = '0',
                        help = ('Defines how much l1 constraint to use.'))

    parser.add_argument('--l2_reg', '-l2', action = 'store',
                        required = False,
                        # default = 1e-5,
                        default = 0,
                        help = ('Defines how much l2 constraint to use.'))

    parser.add_argument('--batch_size', '-bs', action = 'store',
                        required = False,
                        default = '1000',
                        help = ('Defines how many epochs to perform.'))

    parser.add_argument('--epochs', '-ep', action = 'store',
                        required = False,
                        default = '100',
                        help = ('Defines how many epochs to perform.'))
    
    parser.add_argument('--save_path', '-sp', action = 'store',
                        required = False,
                        default = 'max_acts.npy',
                        help = ('path to save model (must be a numpy extension).'))
    
    args = parser.parse_args()
        
    if args.train:
        weight_dict = np.load(args.weight_path).item()
        bias_dict = np.load(args.bias_path).item()
        new_weights = train(weights = np.asarray(list(weight_dict.values())),
                            bias = np.asarray(list(bias_dict.values())),
                            lr = float(args.learning_rate),
                            l1_reg = float(args.l1_reg),
                            l2_reg = float(args.l2_reg),
                            batch_size = int(args.batch_size),
                            epochs = int(args.epochs))
            
        new_weight_dict = {}
        print(new_weights.shape)
        for index, word in enumerate(list(weight_dict)):
            new_weight_dict[word] = new_weights[index,:]
        
        np.save(args.save_path, new_weight_dict)         
    elif args.get_weights:
        buildWeights()
    else:
        raise Exception('No commands provided. Please choose one of the options.')