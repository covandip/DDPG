import tensorflow
from keras.layers import Dense, Input, merge
from keras.models import Model

class Critic:
    """
    This class approximates a Q function, Q: (S,A) -> R
    """

    def __init__(self, state_dim, action_dim, params):
        """
        Constructor
        
        :param state_dim: Int for state dimensionality
        :param aciton_dim: Int for action dimensionality
        :param params: tuple for network parameters (layer 1 size, layer 2 size, learning rate)
        """
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._model, self._state_input, self._action_input = self._create_network(params)
        pass


    def _create_network(self, params):
        """
        Docstring
        """

        state_input_layer = Input(shape=[self._state_dim])
        action_input_layer = Input(shape=[self._action_dim])
        s_hidden1 = Dense(params[0], activation = 'relu')(state_input_layer)
        a_hidden1 = Dense(params[0], activation = 'relu')(action_input_layer)
        hidden = merge([s_hidden1, a_hidden1], mode = "sum")
        hidden = Dense(params[1], activation = 'relu')(hidden)
        output_layer = Dense(self._action_dim, activation = 'relu')(hidden)
        model = Model(inputs=[state_input_layer, action_state_layer], outputs=[output_layer])
        model.compile(loss = 'mse', optimizer=Adam(lr=params[2]))
        return model, state_input_layer, action_input_layer


