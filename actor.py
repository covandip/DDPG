import tensorflow
from keras.layers import Input, Dense
from keras.models import Model

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
        :param params: Tuple for the model parameters, should be (layer1 size, layer2 size)
        """
        self._session = tf_session
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        # create actor model
        self._model, self._model_weights, self._model_input = self._initialize_network(params)

        # create target model
        self._target_model, self._target_weights, self._target_input = self._initialize_network(params)

        # create gradients for training
        self._q_gradients = tensorflow.placeholder(tensorflow.float32, [None, self._action_dim])
        self._param_gradients = tensorflow.gradients(self._model.output, self._model_weights, -self._q_gradients)
        self._gradients = zip(self._param_gradients, self._model_weights)
        
        # create optimizer
        self._optimizer = tensorflow.train.AdamOptimizer(params[2]).apply_gradients(self._gradients)
        
        # initialize session
        self._session.run(tensorflow.global_variables_initializer())

    def _initialize_network(self, params):
        """
        Creates the actor network
        """

        inputs = Input(shape=[self._state_dim])
        hidden = Dense(params[0], activation = 'relu')(inputs)
        hidden = Dense(params[1], activation = 'relu')(hidden)
        outputs = Dense(self._action_dim, activation = 'sigmoid')(hidden2)
        model = Model(inputs = inputs, outputs =outputs)
        return model, model.weights, inputs

    def train(self, state_sequence, q_gradients):
        self_session.run(self._optimizer, feed_dict={
            self._q_gradients:q_gradients,
            self._model_input:state_sequence
            })



