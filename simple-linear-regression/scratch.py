import numpy as np
import random

# input : surfaces en m²
X = np.array([30, 45, 60, 80, 100, 120, 150])

# output : prix en euros
y = np.array([120000, 180000, 240000, 320000, 400000, 480000, 600000])

# Normalisation (pour ne pas avoir de overflow par la suite)
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

# Initialisation
slope = random.random()
intercept = random.random()
learning_rate = 0.01
epochs = 1000

# f(x) = 𝑤 * x + 𝑏 (https://en.wikipedia.org/wiki/Linear_regression)
def model(X, slope, intercept):
    return slope * X + intercept

# fonction de coût quadratique (https://en.wikipedia.org/wiki/Mean_squared_error#:~:text=In%20statistics%2C%20the%20mean%20squared,values%20and%20the%20true%20value.)
def loss_function(Y_true, Y_pred):
    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    return (1/(2*len(Y_true))) * np.sum((Y_true - Y_pred) ** 2)

# Descente de gradient (https://en.wikipedia.org/wiki/Gradient_descent#:~:text=Gradient%20descent%20is%20a%20method,minimizing%20a%20differentiable%20multivariate%20function.)
def update_parameters(X, y, slope, intercept, learning_rate):
    N = len(X)
    y_pred = slope * X + intercept
    # Calcul des gradients
    dw = - (1/N) * np.sum((y - y_pred) * X)
    db = - (1/N) * np.sum(y - y_pred)
    # Mise à jour des paramètres
    slope = slope - learning_rate * dw
    intercept = intercept - learning_rate * db
    return slope, intercept

if __name__ == "__main__":
    for epoch in range(epochs):
        slope, intercept = update_parameters(X_norm, y_norm, slope, intercept, learning_rate)
        if epoch % 100 == 0:
            y_pred = slope * X_norm + intercept
            loss = loss_function(y_norm, y_pred)
            print(f"Epoch {epoch}, Loss: {loss}, Slope: {slope}, Intercept: {intercept}")

    # Pour prédire le prix d'une surface réelle
    surface = 35
    surface_norm = (surface - X_mean) / X_std
    prix_norm = slope * surface_norm + intercept
    prix = prix_norm * y_std + y_mean
    print(f"Prix prédit pour {surface} m² : {prix:.0f} €")