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
        :param params: Tuple for the model parameters, should be (layer1 size, layer2 size)
        """
        self._tf_session = tf_session
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        # create actor model
        self._model, self._model_weights, self._model_input = self._initialize_network(params)

        # create target model
        self._target_model, self._target_weights, self._target_input = self._initialize_network(params)

        # create gradients for training
        self._action_gradients = tensorflow.placeholder(tensorflow.float32, [None, self._action_dim])
        self._param_gradients = tensorflow.gradients(self._model.output, self._model_weights, -self._action_gradients)
        self._gradients = zip(self._action_gradients, self._param_gradients)
        
        # create optimizer
        self._optimizer = tensorflow.train.AdamOptimizer(param[2]).apply_gradients(self._gradients)
        
        # initialize session
        self._tf_session.run(tensorflow.initialize_all_variables())


    def _initialize_network(self, params):
        """
        Creates the actor network
        """

        inputs = Input(shape=[self._state_dim])
        hidden1 = Dense(params[0], activation = 'relu')(inputs)
        hidden2 = Dense(params[1], activation = 'relu')(hidden1)
        outputs = Dense(self._action_dim, activation = 'sigmoid')(hidden2)
        model = Model(input = inputs, output=outputs)
        return model, model.weights, inputs


    def train(self, state_trajectory, )
