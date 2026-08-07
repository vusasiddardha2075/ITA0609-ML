from sklearn.linear_model import LinearRegression

# Dataset
X = [
    [1000], [2000], [3000], [4000], [5000],
    [6000], [7000], [8000], [9000], [10000],
    [11000], [12000], [13000], [14000], [15000],
    [16000], [17000], [18000], [19000], [20000]
]

y = [
    50, 90, 135, 180, 225,
    270, 315, 360, 410, 460,
    510, 565, 620, 680, 740,
    805, 870, 940, 1015, 1090
]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict calories burned for 12500 steps
prediction = model.predict([[12500]])

# Display output
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted Calories Burned:", prediction[0])
