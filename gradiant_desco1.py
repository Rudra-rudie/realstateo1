import numpy as np

def gradient_descent(X, y):
    m_curr = b_curr = 0
    iterations = 10
    learning_rate = 0.01  # Fixed: positive learning rate and correct variable name
    
    for i in range(iterations):
        y_predicted = m_curr * X + b_curr
        cost = (1 / len(X)) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / len(X)) * sum(X * (y - y_predicted))
        bd = -(2 / len(X)) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md  # Fixed: proper update
        b_curr = b_curr - learning_rate * bd  # Fixed: proper update
        print(f"m: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {i+1}")
    
    return m_curr, b_curr

# Fixed: use np.array() with parentheses, not brackets
X = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

# Fixed: correct function name (gradient_descent, not gardiant_descent)
m_final, b_final = gradient_descent(X, y)
print(f"\nFinal values: m {m_final}, b {b_final}")