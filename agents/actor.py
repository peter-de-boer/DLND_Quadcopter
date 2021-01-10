from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, action_low_all, action_high_all):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.action_low_all = action_low_all
        self.action_high_all = action_high_all
        self.action_range_all = self.action_high_all - self.action_low_all

        # Initialize any other variables here

        self.build_model()
        
    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        # initialize with small weights
        
        rn1 = 1./np.sqrt(self.state_size)
        rn2 = 1./np.sqrt(32)
        rn3 = 1./np.sqrt(64)
        
#        net = layers.Dense(units=32, activation='linear', 
#             kernel_initializer=initializers.random_uniform(minval=-rn1,maxval=rn1))(states)
#        net = layers.LeakyReLU(alpha=.1)(net)
#        net = layers.BatchNormalization()(net)
#        net = layers.Dense(units=64, activation='linear', 
#             kernel_initializer=initializers.random_uniform(minval=-rn2,maxval=rn2))(net)
#        net = layers.LeakyReLU(alpha=.1)(net)
#        net = layers.BatchNormalization()(net)
#        net = layers.Dense(units=32, activation='linear', 
#             kernel_initializer=initializers.random_uniform(minval=-rn3,maxval=rn3))(net)
#        net = layers.LeakyReLU(alpha=.1)(net)
#        net = layers.BatchNormalization()(net)
        
        net = layers.Dense(units=64, activation='linear')(states)
        net = layers.LeakyReLU(alpha=.01)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dense(units=128, activation='linear')(net)
        net = layers.LeakyReLU(alpha=.01)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dense(units=64, activation='linear')(net)
        net = layers.LeakyReLU(alpha=.01)(net)
        net = layers.BatchNormalization()(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions1 = layers.Dense(units=self.action_size-1, activation='sigmoid',
            name='raw_actions1', kernel_initializer=initializers.random_uniform(minval=-.0003,maxval=0.0003),
                                  kernel_regularizer=regularizers.l2(0.01))(net)
        
        raw_actions2 = layers.Dense(units=1, activation='sigmoid',
            name='raw_actions2', kernel_initializer=initializers.random_uniform(minval=-.0003,maxval=0.0003),
                                  kernel_regularizer=regularizers.l2(0.01))(net)

        def scale(x):
        #Scale [0, 1] output for each action dimension to proper range
            y=list(x)
            y[:self.action_size]=(x[:self.action_size] * self.action_range) + self.action_low
            y[-1] = (x[-1] * self.action_range_all) + self.action_low_all
            return tuple(y)

          
        actions1 = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions1')(raw_actions1)
        actions2 = layers.Lambda(lambda x: (x * self.action_range_all) + self.action_low_all,
            name='actions2')(raw_actions2)
        # Scale [0, 1] output for each action dimension to proper range
        #actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
        #    name='actions')(raw_actions)
        #actions = layers.Lambda(scale, name='actions')(raw_actions)
        
        actions = layers.concatenate([actions1,actions2])


        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)


        # Incorporate any additional losses here (e.g. from regularizers)
        

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0000005)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)