import numpy as np
import pprint as pp


def get_ABc(l, x0):
    A = 2 * np.array([
        x0.T - l[0].T, 
        x0.T - l[1].T, 
        x0.T - l[2].T
    ])

    B = -np.eye(3)

    x0Tx0 = np.matmul(x0, x0)
    c = np.array([
        x0Tx0 - np.matmul(l[0].T, l[0]),
        x0Tx0 - np.matmul(l[1].T, l[1]),
        x0Tx0 - np.matmul(l[2].T, l[2])
    ])

    return (A, B, c)


def get_M_b(A, B, c, y_bar):

    M1 = np.hstack([np.zeros((2,2)), np.zeros((2, 3)), A.T])
    M2 = np.hstack([np.zeros((3,2)), 2 * np.eye((3)), B.T])
    M3 = np.hstack([A, B, np.zeros((3,3))])

    b1 = np.zeros((1, 2))
    b2 = 2 * y_bar
    b3 = c

    M = np.concatenate((M1, M2, M3), axis=0)
    b = np.concatenate((b1,b2, b3), axis=None)

    return (M, b)


def calculate_x(noisy_d=False, n_steps=1):
    print('\nCalculating x with noisy_d={} and n_steps={}'.format(noisy_d, n_steps))

    l = np.array([[5,25],[32,22],[29,5]])
    x0 = l.mean(axis=0)
    
    d_bar = np.array([17, 13, 15]) + noisy_d*np.random.randn((3))
    y_bar = np.square(d_bar)

    for i in range(n_steps):
        (A, B, c) = get_ABc(l, x0)
        (M, b) = get_M_b(A, B, c, y_bar)

        res = np.linalg.solve(M, b)
        x = res[:2]
        print('    Iteration {}: x={}'.format(i, x))

        x0 = x

calculate_x(noisy_d=False, n_steps=1)
calculate_x(noisy_d=True, n_steps=20)