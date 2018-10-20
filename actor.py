import tensorflow
import keras

class Actor:
    """
    This class will approximate an action policy A: S -> A
    """
    def __init__(self, tf_session, state_dim, action_dim, params):
        """
        Constructor for the actor network

        :param tf_session: the tensorflow session
        :param state_dim: Int for the state dimensionality
        :param action_dim: Int for the action dimensionality
        :param params: Tuple for the model parameters
        """
        self._tf_session = tf_session
        self._state_dim = state_dim
        self._action_dim = action_dim


    def _initialize_network(self, params):
        """
        Creates the actor network
        """

        inputs = Input(shape=[self._state_dim])
        hidden1 = Dense(params[0], activation = 'relu')(inputs)
        hidden2 = Dense(params[1], activation = 'relu')(hidden1)
        outputs = Dense(self._action_dim, activation = 'sigmoid')(hidden2)
        model = Model(input = inputs, output=outputs)
        return model, model.trainable_weights, inputs

