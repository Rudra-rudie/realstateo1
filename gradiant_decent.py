x=np.array[1,2,3,4,5]
import numpy as np


def gradient_descent(X, y, learning_rate=0.001, iterations=1000):
    """Simple linear regression using gradient descent.

    Returns (m, b, cost) after `iterations`.
    """
    m_curr = 0.0
    b_curr = 0.0
    n = len(X)

    for i in range(iterations):
        y_predicted = m_curr * X + b_curr
        cost = (1 / n) * np.sum((y - y_predicted) ** 2)
        md = -(2 / n) * np.sum(X * (y - y_predicted))
        bd = -(2 / n) * np.sum(y - y_predicted)

        # Update parameters (standard gradient descent step)
        m_curr -= learning_rate * md
        b_curr -= learning_rate * bd

    return m_curr, b_curr, cost


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 7, 9, 11, 13], dtype=float)

    m, b, cost = gradient_descent(x, y, learning_rate=0.01, iterations=1000)
    print(f"m: {m:.4f}, b: {b:.4f}, cost: {cost:.6f}")