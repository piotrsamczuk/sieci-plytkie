import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # He
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # Momentum i Adam
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.mW1, self.mb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.mW2, self.mb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        
        self.learning_rate = 0.001
        self.lambda_reg = 0.01  # Współczynnik regularyzacji L2
        
        self.mse_errors = []
        self.classification_errors = []
        self.W1_history = []
        self.W2_history = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.error = output - y
        self.delta2 = self.error * self.sigmoid_derivative(output)
        self.dW2 = np.dot(self.a1.T, self.delta2) + self.lambda_reg * self.W2  # Regularyzacja L2
        self.db2 = np.sum(self.delta2, axis=0, keepdims=True)
        
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.a1)
        self.dW1 = np.dot(X.T, self.delta1) + self.lambda_reg * self.W1  # Regularyzacja L2
        self.db1 = np.sum(self.delta1, axis=0)
    
    def update_weights_adam(self, t):
        # Aktualizacja dla W1 i b1
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * self.dW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (self.dW1 ** 2)
        mW1_hat = self.mW1 / (1 - self.beta1 ** t)
        vW1_hat = self.vW1 / (1 - self.beta2 ** t)
        self.W1 -= self.learning_rate * mW1_hat / (np.sqrt(vW1_hat) + self.epsilon)

        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * self.db1
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (self.db1 ** 2)
        mb1_hat = self.mb1 / (1 - self.beta1 ** t)
        vb1_hat = self.vb1 / (1 - self.beta2 ** t)
        self.b1 -= self.learning_rate * mb1_hat / (np.sqrt(vb1_hat) + self.epsilon)

        # Aktualizacja dla W2 i b2
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * self.dW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (self.dW2 ** 2)
        mW2_hat = self.mW2 / (1 - self.beta1 ** t)
        vW2_hat = self.vW2 / (1 - self.beta2 ** t)
        self.W2 -= self.learning_rate * mW2_hat / (np.sqrt(vW2_hat) + self.epsilon)

        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * self.db2
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (self.db2 ** 2)
        mb2_hat = self.mb2 / (1 - self.beta1 ** t)
        vb2_hat = self.vb2 / (1 - self.beta2 ** t)
        self.b2 -= self.learning_rate * mb2_hat / (np.sqrt(vb2_hat) + self.epsilon)
    
    def calculate_mse(self, X, y):
        predictions = self.forward(X)
        return np.mean(np.square(predictions - y))
    
    def calculate_classification_error(self, X, y):
        predictions = self.forward(X)
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes != y)
    
    def train(self, X, y, epochs=10000, batch_size=2, error_threshold=0.001):
        for epoch in range(1, epochs + 1):
            # Mini-batch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
                self.update_weights_adam(epoch)
            
            # Zbieranie danych do wykresów
            mse_error = self.calculate_mse(X, y)
            classification_error = self.calculate_classification_error(X, y)
            self.mse_errors.append(mse_error)
            self.classification_errors.append(classification_error)
            self.W1_history.append(self.W1.copy())
            self.W2_history.append(self.W2.copy())
            
            # Sprawdzenie błędu
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {mse_error:.6f}')
                if mse_error < error_threshold:
                    print(f"Training stopped early at epoch {epoch} with loss {mse_error:.6f}")
                    return
    
    def plot_results(self):
        # Wykres błędu MSE
        plt.figure(figsize=(12, 6))
        plt.plot(self.mse_errors, label='MSE Error')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE Error Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

        # Wykres błędu klasyfikacji
        plt.figure(figsize=(12, 6))
        plt.plot(self.classification_errors, label='Classification Error', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Error')
        plt.title('Classification Error Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

        # Wykres wag W1
        plt.figure(figsize=(12, 6))
        for i in range(self.W1.shape[1]):
            plt.plot([w[0, i] for w in self.W1_history], label=f'W1[{i}]')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.title('Weights W1 Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

        # Wykres wag W2
        plt.figure(figsize=(12, 6))
        plt.plot([w[0, 0] for w in self.W2_history], label='W2')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.title('Weights W2 Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)
nn.train(X, y, epochs=10000, batch_size=2, error_threshold=0.001)

print("Predictions:")
for i in range(X.shape[0]):
    print(f"Input: {X[i]}, Predicted: {nn.forward(X[i:i+1])}")

# Generowanie wykresów
nn.plot_results()