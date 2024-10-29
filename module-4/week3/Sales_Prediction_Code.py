
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Question 1: Completing the predict function in the CustomLinearRegression class
class CustomLinearRegression:
    def __init__(self, X_data, y_target, learning_rate=0.01, num_epochs=10000):
        self.num_samples = X_data.shape[0]
        self.X_data = np.c_[np.ones((self.num_samples, 1)), X_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.theta = np.random.randn(self.X_data.shape[1], 1)
        self.losses = []

    def compute_loss(self, y_pred, y_target):
        loss = np.mean((y_pred - y_target) ** 2)  # Sample code for loss calculation
        return loss

    def predict(self, X_data):
        y_pred = X_data.dot(self.theta)  # Answer to Question 1
        return y_pred

    def fit(self):
        for epoch in range(self.num_epochs):
            y_pred = self.predict(self.X_data)  # Step C
            loss = self.compute_loss(y_pred, self.y_target)  # Step B
            self.losses.append(loss)
            loss_grd = 2 * (y_pred - self.y_target) / self.num_samples  # Step A
            gradients = self.X_data.T.dot(loss_grd)
            self.theta = self.theta - self.learning_rate * gradients  # Step D
            if epoch % 50 == 0:
                print(f'Epoch: {epoch} - Loss: {loss}')
        return {
            'loss': sum(self.losses) / len(self.losses),
            'weight': self.theta
        }

# Question 3: R2 Score function
def r2score(y_pred, y):
    rss = np.sum((y_pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (rss / tss)  # Answer to Question 3
    return r2

# Question 4: Testing r2score function with given cases
y_pred_case1 = np.array([1, 2, 3, 4, 5])
y_case1 = np.array([1, 2, 3, 4, 5])
r2_case1 = r2score(y_pred_case1, y_case1)

y_pred_case2 = np.array([1, 2, 3, 4, 5])
y_case2 = np.array([3, 5, 5, 2, 4])
r2_case2 = r2score(y_pred_case2, y_case2)

print(f'R2 Score Case 1: {r2_case1}, R2 Score Case 2: {r2_case2}')

# Question 7: Polynomial Features Function
def create_polynomial_features(X, degree=2):
    X_new = X
    for d in range(2, degree + 1):
        X_new = np.c_[X_new, np.power(X, d)]
    return X_new  # Answer to Question 7

# Question 8: Polynomial features for multiple inputs
def create_polynomial_features_multiple(X, degree=2):
    X_mem = []
    for X_sub in X.T:
        X_sub = X_sub.T
        X_new = X_sub
        for d in range(2, degree + 1):
            X_new = np.c_[X_new, np.power(X_sub, d)]
        X_mem.append(X_new.T)
    return np.c_[X_mem].T  # Answer to Question 8

# Question 9: One Hot Encoding Example
df = pd.DataFrame({'Influencer': ['Macro', 'Mega', 'Micro', 'Nano']})
df_encoded = pd.get_dummies(df)  # Answer to Question 9
print("One-Hot Encoded DataFrame:\n", df_encoded)

# Question 10: StandardScaler and Polynomial Features
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Approximate scaler mean output
scaler_mean = scaler.mean_[0]  # Answer to Question 10
print("Scaler mean approximation:", scaler_mean)

# Polynomial features for scaled data
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# Linear Regression model training
model = LinearRegression()
model.fit(X_poly, np.array([1, 2, 3]))

# Prediction and R2 Score
preds = model.predict(X_poly)
r2 = r2_score(np.array([1, 2, 3]), preds)
print("Model R2 score:", r2)
