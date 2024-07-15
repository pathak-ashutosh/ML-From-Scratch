import pandas as pd
import matplotlib.pyplot as plt

def calculate_loss(m, b, dataframe):
    n = len(dataframe)
    total_error = 0
    error = 0
    
    for i in range(n):
        x_i = dataframe.loc[i, 'age']
        y_i = dataframe.loc[i, 'charges']
        error += (y_i - (m * x_i + b))**2   # mean-squared error
    
    total_error = error / float(n)  # E = 1/n * sum((y_i - (m * x_i + b))**2)
    return total_error

def gradient_descent(current_m, current_b, dataframe, learning_rate):
    gradient_m = 0
    gradient_b = 0
    n = len(dataframe)
    
    for i in range(n):
        x_i = dataframe.iloc[i].age     #  because age has a clear trend vs charges
        y_i = dataframe.iloc[i].charges
        gradient_m += x_i * (y_i - (current_m * x_i + current_b))
        gradient_b += (y_i - (current_m * x_i + current_b))
    
    gradient_m *= -2/n  # dE/dm = -(2/n) * sum(x_i * (y_i - (current_m * x_i + current_b)))
    gradient_b *= -2/n  # dE/db = -(2/n) * sum(y_i - (current_m * x_i + current_b))
    
    m = current_m - gradient_m*learning_rate
    b = current_b - gradient_b*learning_rate
    
    return m, b

if __name__ == '__main__':
    # Dataset taken from https://www.kaggle.com/datasets/mirichoi0218/insurance
    filepath = "../data/insurance.csv"
    df = pd.read_csv(filepath)

    m = 0
    b = 0
    learning_rate = 0.0001
    epochs = 300
    
    for i in range(epochs):
        if i % 50 == 0:
            print(f"Epoch: {i}")
            print(f"Loss: {calculate_loss(m, b, df)}")
        m, b = gradient_descent(m, b, df, learning_rate)
    
    print(f"Optimal m and b: {m}, {b}")
    print(f"Final Loss: {calculate_loss(m, b, df)}")
    
    min_age = int(df.age.min())
    max_age = int(df.age.max())
    
    plt.scatter(df.age, df.charges, color='black')
    plt.plot(list(range(min_age, max_age)), [m*x + b for x in range(min_age, max_age)], color='red')
    plt.show()
    