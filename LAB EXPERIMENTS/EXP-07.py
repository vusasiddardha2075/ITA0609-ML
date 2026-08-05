import math

# Dataset: Study Hours and Result
X = [1, 2, 3, 4, 5, 6, 7, 8]
Y = [0, 0, 0, 0, 1, 1, 1, 1]

# Initialize weights and bias
w = 0
b = 0

learning_rate = 0.1
epochs = 1000

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Training the Logistic Regression Model
for epoch in range(epochs):
    dw = 0
    db = 0

    for i in range(len(X)):
        z = w * X[i] + b
        y_pred = sigmoid(z)

        dw += (y_pred - Y[i]) * X[i]
        db += (y_pred - Y[i])

    w = w - learning_rate * dw / len(X)
    b = b - learning_rate * db / len(X)

# Prediction
print("Study Hours\tActual\tPredicted")

correct = 0

for i in range(len(X)):
    probability = sigmoid(w * X[i] + b)

    if probability >= 0.5:
        prediction = 1
    else:
        prediction = 0

    print(X[i], "\t\t", Y[i], "\t", prediction)

    if prediction == Y[i]:
        correct += 1

# Accuracy
accuracy = (correct / len(X)) * 100

print("\nAccuracy =", accuracy, "%")
