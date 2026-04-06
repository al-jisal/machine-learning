import numpy as np

class Adaline():
    ''' Single-layer neural network
    Network weights are organized [wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    Bias is stored separately from wts.
    '''
    def __init__(self):
        '''ADALINE Constructor'''
        self.wts = None
        self.b = None
        self.loss_history = None
        self.accuracy_history = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        return None if self.wts is None else self.wts.copy()

    def get_bias(self):
        ''' Returns a copy of the bias'''
        return self.b

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features, wts, bias)

        The formula used is: z = Xw + b
        where X is the feature matrix, w is the weight vector, and b is the bias.

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''
        # We use np.dot for the matrix-vector product.
        # features (N, M) @ wts (M,) results in a vector of (N,)
        return np.dot(features, self.wts) + self.b

    def activation(self, net_in):
        '''Applies the activation function to the net input.
        For ADALINE, this is the identity function: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act: ndarray. Shape = [Num samples N,]
        '''
        # In vanilla ADALINE, activation is just the identity
        return net_in

    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]

        Returns:
        ----------
        Predicted classes (-1 or +1). Shape = [Num samples N,]
        '''
        net_in = self.net_input(features)
        net_act = self.activation(net_in)
        # If activation >= 0, predict +1. Otherwise, predict -1.
        return np.where(net_act >= 0.0, 1, -1)

    def accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
        y_pred: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        float. The accuracy (proportion of correct predictions)
        '''
        return np.mean(y == y_pred)

    def loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss

        Formula: 0.5 * sum((y - net_act)^2)
        Note: The 0.5 is a common convention in ADALINE to make the 
        derivative cleaner during the weight update step.

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
        net_act: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        float. The SSE loss.
        '''
        # Calculate errors: (target - activation)
        errors = y - net_act
        sse = np.sum(errors**2)
        return 0.5 * sse

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        
        The gradients are the partial derivatives of the SSE loss:
        dLoss/dw = -sum(errors * features)
        dLoss/db = -sum(errors)
        '''
        # Gradient for weights: Negative dot product of features and errors
        # Shape: (M, N) dot (N,) -> (M,)
        grad_wts = -np.dot(features.T, errors)
        grad_bias = -np.sum(errors)
        
        return grad_bias, grad_wts

    def fit(self, features, y, n_epochs=1000, lr=0.001, r_seed=None):
        '''Trains the network using Batch Gradient Descent'''
        rng = np.random.default_rng(r_seed)
        n_features = features.shape[1]
        self.wts = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = 0.0
        
        self.loss_history = []
        self.accuracy_history = []

        # Main training loop
        for epoch in range(n_epochs):
            net_in = self.net_input(features)
            net_act = self.activation(net_in)
            errors = y - net_act
            
            # Compute and store performance metrics
            epoch_loss = self.loss(y, net_act)
            y_pred = self.predict(features)
            epoch_acc = self.accuracy(y, y_pred)
            
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_acc)
            
            # Backward Pass (Backpropagation)
            grad_bias, grad_wts = self.gradient(errors, features)
            
            # Update weights and bias: move opposite the gradient
            self.wts -= lr * grad_wts
            self.b -= lr * grad_bias
            
        return self.loss_history, self.accuracy_history
