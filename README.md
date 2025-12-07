# MiniNumPy
This project designs a simplified version of NumPy, including core data structures, array operations, and linear algebra.


# Part 1: Core Array Class

a class Array that wraps a Python list

Store attributes: .data,.shape,.ndim,.size

reshape(new_shape), transpose(), __str__ for pretty printing

# Part 2: Array Creation

array(list_or_nested_list)

zeros(shape)

ones(shape)

eye(n)

arrange(start, stop, step)

linspace(start, stop, num)

# Part 3: Elementwise Operations

Overload Python operations(+, -, /, **)

elementwise functions: exp, log, sqrt, abs

reductions: sum, mean, min, max, argmin, argmax

# Part 4: Linear Algebra Module (minilinalg)

matrix/vector operations: dot(a, b), matmul(a, b), norm(a)

basic factorizations/solvers: det(a), inv(a), eig(a)
