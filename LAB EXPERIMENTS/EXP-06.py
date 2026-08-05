# Training data
X_train = [
    ["Sunny", "Hot"],
    ["Sunny", "Hot"],
    ["Overcast", "Hot"],
    ["Rainy", "Mild"],
    ["Rainy", "Cool"],
    ["Rainy", "Cool"],
    ["Overcast", "Cool"],
    ["Sunny", "Mild"],
    ["Sunny", "Cool"],
    ["Rainy", "Mild"]
]

y_train = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]

# Test data
X_test = [
    ["Sunny", "Cool"],
    ["Rainy", "Cool"],
    ["Overcast", "Hot"],
    ["Sunny", "Hot"]
]

y_test = ["Yes", "Yes", "Yes", "No"]

# Find unique class labels
classes = list(set(y_train))

# Calculate prior probabilities
priors = {}
for c in classes:
    priors[c] = y_train.count(c) / len(y_train)

# Predict function
def predict(sample):
    probabilities = {}

    for c in classes:
        prob = priors[c]

        class_rows = []
        for i in range(len(y_train)):
            if y_train[i] == c:
                class_rows.append(X_train[i])

        for j in range(len(sample)):
            count = 0
            for row in class_rows:
                if row[j] == sample[j]:
                    count += 1

            # Laplace smoothing
            prob *= (count + 1) / (len(class_rows) + len(set([r[j] for r in X_train])))

        probabilities[c] = prob

    return max(probabilities, key=probabilities.get)

# Predict all test samples
predictions = []
for sample in X_test:
    predictions.append(predict(sample))

# Display predictions
print("Actual Labels    :", y_test)
print("Predicted Labels :", predictions)

# Confusion Matrix
labels = classes
matrix = []

for actual in labels:
    row = []
    for predicted in labels:
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == actual and predictions[i] == predicted:
                count += 1
        row.append(count)
    matrix.append(row)

print("\nConfusion Matrix")
print("      ", labels)
for i in range(len(labels)):
    print(labels[i], matrix[i])

# Accuracy
correct = 0
for i in range(len(y_test)):
    if y_test[i] == predictions[i]:
        correct += 1

accuracy = (correct / len(y_test)) * 100

print("\nAccuracy = ", accuracy, "%")
