import numpy as np


# 实现一个简单与门
def AND(x1, x2) -> int:
    w1, w2, theta = 1, 1, 1
    res = x1 * w1 + x2 * w2
    if res <= theta:
        return 0
    return 1


def AND(x1, x2):
    X = np.array([x1, x2])
    w = np.array([1, 1])
    b = -1
    res = np.sum(X * w) + b
    if res <= 0:
        return 0
    return 1


print(AND(0, 0))  # 0
print(AND(1, 0))  # 0
print(AND(0, 1))  # 0
print(AND(1, 1))  # 1
