import numpy as np


# class MSELoss:

#     def forward(self, A, Y):
#         """
#         Calculate the Mean Squared error
#         :param A: Output of the model of shape (N, C)
#         :param Y: Ground-truth values of shape (N, C)
#         :Return: MSE Loss(scalar)

#         """

#         self.A = A
#         self.Y = Y
#         self.N = A.shape[0]  # TODO
#         self.C = A.shape[1]  # TODO
#         se = (A - Y)*(A-Y)  # TODO
#         sse = self.N*se*self.C  # TODO
#         mse = sse / (2*self.N*self.C)  # TODO

#         return mse

#     def backward(self):

#         dLdA = (2 / self.N) * (self.A - self.Y)

#         return dLdA

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape

        
        SE = (A - Y) ** 2

        
        SSE = np.sum(SE)

        mse = SSE / (2*self.N * self.C)

        return mse

    def backward(self):
        """
        Calculate the gradient of the loss with respect to A
        :Return: Gradient of loss with respect to A
        """
        dLdA = (self.A - self.Y) / (self.N * self.C)

        return dLdA


# class CrossEntropyLoss:

#     def forward(self, A, Y):
#         """
#         Calculate the Cross Entropy Loss
#         :param A: Output of the model of shape (N, C)
#         :param Y: Ground-truth values of shape (N, C)
#         :Return: CrossEntropyLoss(scalar)

#         Refer the the writeup to determine the shapes of all the variables.
#         Use dtype ='f' whenever initializing with np.zeros()
#         """
#         self.A = A
#         self.Y = Y
#         N,C = A.shape  # TODO
        
#         Ones_C = np.ones((C,1),dtype = 'f')  # TODO
#         Ones_N = np.ones((N,1),dtype = 'f')  # TODO

#         exp_A = np.exp(A - np.max(A , axis = 1, keepdims=True))
#         self.softmax = exp_A / np.sum(exp_A , axis=1 , keepdims=True)  # TODO
#         crossentropy = -Y * np.log(self.softmax + 1e-9 )  # TODO
#         sum_crossentropy = np.dot(crossentropy,Ones_C)  # TODO
#         L =np.dot(Ones_N.T, sum_crossentropy) / N

#         return L

#     def backward(self):

#         dLdA = (self.A - self.Y) / (self.N*(self.C))  # TODO

#         return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)
        """
        self.A = A
        self.Y = Y
        N, C = A.shape  # Number of samples and classes

        # Initialize Ones vectors
        Ones_C = np.ones((C, 1), dtype='f')  # Column vector of ones with shape (C, 1)
        Ones_N = np.ones((N, 1), dtype='f')  # Column vector of ones with shape (N, 1)

        # Compute softmax
        exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))  # For numerical stability
        self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)

        # Compute cross-entropy for each sample
        crossentropy = -np.sum(Y * np.log(self.softmax + 1e-9), axis=1, keepdims=True)  # Shape (N, 1)

        # If you want to use Ones_C for summing over classes (not typically needed here), you could do:
        # crossentropy = -np.dot(Y * np.log(self.softmax + 1e-9), Ones_C)  # This would also result in shape (N, 1)

        # Sum cross-entropy losses across the batch using Ones_N
        sum_crossentropy = np.dot(Ones_N.T, crossentropy)  # Result is a scalar

        # Average cross-entropy loss
        L = sum_crossentropy / N

        return L

    def backward(self):
        """
        Compute the gradient of the Cross-Entropy Loss with respect to A
        """
        # N = self.A.shape[0]
        # dLdA = (self.softmax - self.Y) / N  # Gradient of loss with respect to A
        dLdA = self.softmax-self.Y

        return dLdA