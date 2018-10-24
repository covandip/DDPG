import tensorflow
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

class Critic:
    """
    This class approximates a Q function, Q: (S,A) -> R
    """

    def __init__(self, tf_session, tau, state_dim, action_dim, params):
        """
        Constructor
        
        :param state_dim: Int for state dimensionality
        :param aciton_dim: Int for action dimensionality
        :param params: tuple for network parameters (layer 1 size, layer 2 size, learning rate)
        """
        self._tau = tau
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._sess = tf_session
        self._model, self._state_input, self._action_input = self._create_network(params)
        self._target_model, _, _  = self._create_network(params)
        pass


    def _create_network(self, params):
        """
        Docstring
        """

        state_input_layer = Input(shape=[self._state_dim])
        action_input_layer = Input(shape=[self._action_dim])
        s_hidden1 = Dense(params[0], activation = 'relu')(state_input_layer)
        a_hidden1 = Dense(params[0], activation = 'relu')(action_input_layer)
        hidden = concatenate([s_hidden1, a_hidden1])
        hidden = Dense(params[1], activation = 'relu')(hidden)
        output_layer = Dense(self._action_dim, activation = 'sigmoid')(hidden)
        model = Model(inputs=[state_input_layer, action_input_layer], outputs=[output_layer])
        model.compile(loss = 'mse', optimizer=Adam(lr=params[2]))
        return model, state_input_layer, action_input_layer

    def train(self, states, actions, q_targets):
        self._model.train_on_batch([states, actions], q_targets)
    
    def update_target_network(self):
        critic_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        
        new_weights = [self._tau*critic_weight + (1-self._tau)*target_weight
                       for critic_weight, target_weight in
                       zip(critic_weights, target_weights)]
        self._target_model.set_weights(new_weights)
        
        