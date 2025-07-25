import numpy as np
import random

# X : [surface, nombre de chambres]
X = np.array([
    [30, 1],
    [45, 2],
    [60, 2],
    [80, 3],
    [100, 4],
    [120, 4],
    [150, 5]
])

# output : prix
y = np.array([120000, 180000, 240000, 320000, 400000, 480000, 600000])

# Normalisation (pour ne pas avoir de overflow par la suite)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

# Initialisation
weights = np.random.random(X.shape[1])  # un poids par feature
intercept = random.random()
learning_rate = 0.01
epochs = 1000

# f(x) = 𝑤 * x + 𝑏 (https://en.wikipedia.org/wiki/Linear_regression)
def model(X, weights, intercept):
    return np.dot(X, weights) + intercept

# fonction de coût quadratique (https://en.wikipedia.org/wiki/Mean_squared_error#:~:text=In%20statistics%2C%20the%20mean%20squared,values%20and%20the%20true%20value.)
def loss_function(Y_true, Y_pred):
    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    return (1/(2*len(Y_true))) * np.sum((Y_true - Y_pred) ** 2)

# Descente de gradient (https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20method,minimizing%20a%20differentiable%20multivariate%20function.)
def update_parameters(X, y, weights, intercept, learning_rate):
    N = len(X)
    y_pred = model(X, weights, intercept)
    
    # Gradients pour chaque poids
    dw = - (1/N) * np.dot(X.T, (y - y_pred))
    db = - (1/N) * np.sum(y - y_pred)
    
    # Mise à jour
    weights = weights - learning_rate * dw
    intercept = intercept - learning_rate * db
    return weights, intercept

if __name__ == "__main__":
    for epoch in range(epochs):
        weights, intercept = update_parameters(X_norm, y_norm, weights, intercept, learning_rate)
        if epoch % 100 == 0:
            y_pred = model(X_norm, weights, intercept)
            loss = loss_function(y_norm, y_pred)
            print(f"Epoch {epoch}, Loss: {loss}, Weights: {weights}, Intercept: {intercept}")

    # Pour prédire le prix d'une surface réelle
    new_data = np.array([[35, 2]])  # 35m², 2 chambres
    new_data_norm = (new_data - X_mean) / X_std
    prix_norm = model(new_data_norm, weights, intercept)
    prix = prix_norm * y_std + y_mean
    print(f"Prix prédit pour {new_data[0][0]} m² et {new_data[0][1]} chambres : {prix[0]:.0f} €")