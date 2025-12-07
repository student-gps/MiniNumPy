# mininumpy.py
from typing import Tuple, Union, Callable, List
import math
import warnings

Number = Union[int, float]

# Safe math
def _safe_div(x: Number, y: Number) -> Number:
    """1/0 -> inf, 0/0 -> nan"""
    if y == 0:
        return math.inf if x != 0 else math.nan
    return x / y

def _safe_pow(base: Number, exp: Number) -> Number:
    """ln(<0) -> nan; 0**0 -> 1"""
    # define 0**0 = 1 for this library (numpy does 1.0)
    if base == 0 and exp == 0:
        return 1.0
    # negative base with fractional exponent -> nan (we avoid complex)
    if base < 0 and not float(exp).is_integer():
        warnings.warn(f"({base})**({exp}): complex result, returning nan", RuntimeWarning)
        return math.nan
    return base ** exp

def _safe_log(x: Number) -> Number:
    if x <= 0:
        warnings.warn(f"log({x}): domain error, returning nan", RuntimeWarning)
        return math.nan
    return math.log(x)

def _safe_log10(x: Number) -> Number:
    if x <= 0:
        warnings.warn(f"log10({x}): domain error, returning nan", RuntimeWarning)
        return math.nan
    return math.log10(x)

def _safe_sqrt(x: Number) -> Number:
    if x < 0:
        warnings.warn(f"sqrt({x}): domain error, returning nan", RuntimeWarning)
        return math.nan
    return math.sqrt(x)


# Array class
class Array:
    def __init__(self, data: Union[list, 'Array', Number]):
        if isinstance(data, Array):
            # copy by value (浅复制 data 结构)
            self.data = _deepcopy_list(data.data) if isinstance(data.data, list) else data.data
        else:
            self.data = data
        self.shape = self._compute_shape(self.data)
        self.ndim = len(self.shape)
        self.size = self._compute_size(self.shape)

    # Utility helpers
    def _compute_shape(self, data) -> Tuple[int, ...]:
        shape = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            if len(current) == 0:
                break
            current = current[0]
        return tuple(shape)

    def _compute_size(self, shape: Tuple[int, ...]) -> int:
        if not shape:
            return 1 if not isinstance(self.data, list) else 0
        s = 1
        for x in shape:
            s *= x
        return s

    def _flatten(self, data=None) -> List[Number]:
        if data is None:
            data = self.data
        if isinstance(data, list):
            res = []
            for item in data:
                res.extend(self._flatten(item))
            return res
        else:
            return [data]

    # Reshape / Transpose
    def reshape(self, new_shape: Tuple[int, ...]) -> 'Array':
        # flatten then rebuild nested lists by new_shape
        flat = self._flatten()
        new_size = 1
        for x in new_shape:
            new_size *= x
        if new_size != len(flat):
            raise ValueError(f"Cannot reshape array of size {len(flat)} into shape {new_shape}")

        def build_flat(flat_list, shape):
            if not shape:
                return flat_list.pop(0)
            n = shape[0]
            rest = shape[1:]
            return [build_flat(flat_list, rest) for _ in range(n)]

        flat_copy = flat[:]  # copy, we'll pop from front
        nested = build_flat(flat_copy, new_shape)
        return Array(nested)

    def transpose(self) -> 'Array':
        if self.ndim != 2:
            raise ValueError("transpose only supports 2D arrays")
        transposed = list(map(list, zip(*self.data)))
        return Array(transposed)

    # in-place helpers (less used)
    def _transpose_inplace(self):
        if self.ndim != 2:
            raise ValueError("transpose only supports 2D arrays")
        self.data = list(map(list, zip(*self.data)))
        self.shape = self._compute_shape(self.data)
        self.ndim = len(self.shape)
        self.size = self._compute_size(self.shape)

    # String / printing
    def __str__(self) -> str:
        if not isinstance(self.data, list):
            return f"Array({self.data})"
        # pretty print small arrays
        if self.ndim == 1:
            return f"Array({self.data})"
        if self.ndim == 2:
            rows = [" ".join(map(str, row)) for row in self.data]
            return "[\n " + "\n ".join(rows) + "\n]"
        # fallback
        return f"Array(data={self.data}, shape={self.shape})"

    def __repr__(self) -> str:
        return self.__str__()

    # Elementwise operations with broadcasting-to-scalar
    @staticmethod
    def _recursive_op(data1, data2, operation):
        if isinstance(data1, list):
            if isinstance(data2, list):
                if len(data1) != len(data2):
                    raise ValueError("Cannot perform element-wise operation: array shapes must match.")
                return [Array._recursive_op(a, b, operation) for a, b in zip(data1, data2)]
            else:
                # broadcast scalar to each element of data1
                return [Array._recursive_op(a, data2, operation) for a in data1]
        else:
            if isinstance(data2, list):
                raise ValueError("Cannot broadcast array to scalar position.")
            return operation(data1, data2)

    def _binary_op_wrapper(self, other, operation):
        data2 = other.data if isinstance(other, Array) else other
        if isinstance(other, Array) and self.shape != other.shape:
            # exception: allow scalar-like arrays? we keep it strict
            raise ValueError(f"Operands could not be broadcast together with shapes {self.shape} and {other.shape}")
        new_data = Array._recursive_op(self.data, data2, operation)
        return Array(new_data)

    # arithmetic operators
    def __add__(self, other):   return self._binary_op_wrapper(other, lambda x, y: x + y)
    def __radd__(self, other):  return self.__add__(other)

    def __sub__(self, other):   return self._binary_op_wrapper(other, lambda x, y: x - y)
    def __rsub__(self, other):
        # other - self : implement by broadcasting if other is scalar or Array with same shape
        if isinstance(other, Array):
            return other._binary_op_wrapper(self, lambda x, y: x - y)
        else:
            return Array(other) - self

    def __mul__(self, other):   return self._binary_op_wrapper(other, lambda x, y: x * y)
    def __rmul__(self, other):  return self.__mul__(other)

    def __truediv__(self, other): return self._binary_op_wrapper(other, _safe_div)
    def __rtruediv__(self, other):
        if isinstance(other, Array):
            return other._binary_op_wrapper(self, _safe_div)
        else:
            # scalar / array
            return Array(other) / self

    def __pow__(self, other):   return self._binary_op_wrapper(other, _safe_pow)

    # matrix multiply operator
    def __matmul__(self, other):
        from math import isfinite
        return matmul(self, other)

    # Unary elementwise
    def _unary_op(self, func: Callable[[Number], Number]) -> 'Array':
        def rec(d):
            if isinstance(d, list):
                return [rec(item) for item in d]
            else:
                return func(d)
        return Array(rec(self.data))

    def exp(self): return self._unary_op(math.exp)
    def log(self): return self._unary_op(_safe_log)
    def log10(self): return self._unary_op(_safe_log10)
    def sqrt(self): return self._unary_op(_safe_sqrt)
    def abs(self): return self._unary_op(abs)

    # Reductions
    def sum(self) -> Number:
        def rec(d):
            if isinstance(d, list):
                s = 0
                for item in d:
                    s += rec(item)
                return s
            else:
                return d
        return rec(self.data)

    def mean(self) -> float:
        flat = self._flatten()
        return sum(flat) / len(flat) if len(flat) > 0 else float('nan')

    def max(self) -> Number:
        flat = self._flatten()
        if not flat:
            raise ValueError("max() of empty array")
        return max(flat)

    def min(self) -> Number:
        flat = self._flatten()
        if not flat:
            raise ValueError("min() of empty array")
        return min(flat)

    def argmax(self) -> int:
        flat = self._flatten()
        if not flat:
            raise ValueError("argmax() of empty array")
        m = max(flat)
        return flat.index(m)

    def argmin(self) -> int:
        flat = self._flatten()
        if not flat:
            raise ValueError("argmin() of empty array")
        m = min(flat)
        return flat.index(m)

    # Indexing helpers (simple)
    def tolist(self):
        return self.data

    # Creation helpers (class methods)
    @staticmethod
    def _make_filler(shape: Tuple[int, ...], value: Number):
        if not shape:
            return value
        return [Array._make_filler(shape[1:], value) if False else Array._make_filler(shape[1:], value) for _ in range(shape[0])]

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> 'Array':
        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple of ints")
        data = _build_filler(shape, 0)
        return cls(data)

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> 'Array':
        data = _build_filler(shape, 1)
        return cls(data)

    @classmethod
    def eye(cls, n: int) -> 'Array':
        if n < 0:
            raise ValueError("size must be non-negative")
        data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(data)

    @classmethod
    def arange(cls, start: int, stop: int, step: int = 1) -> 'Array':
        if step == 0:
            raise ValueError("step must be non-zero")
        # Python range stops before stop, keep similar semantics
        data = list(range(start, stop, step))
        return cls(data)

    @classmethod
    def linspace(cls, start: float, stop: float, num: int) -> 'Array':
        if num <= 0:
            raise ValueError("num must be positive")
        if num == 1:
            return cls([start])
        step = (stop - start) / (num - 1)
        data = [start + i * step for i in range(num)]
        return cls(data)

def linspace(start, stop, num):
    return Array.linspace(start, stop, num)

def zeros(shape):
    return Array.zeros(shape)

def ones(shape):
    return Array.ones(shape)

def eye(n):
    return Array.eye(n)

def arange(start, stop, step=1):
    return Array.arange(start, stop, step)

def _build_filler(shape: Tuple[int, ...], val: Number):
    """递归构造多维嵌套 list"""
    if not shape:
        return val
    n = shape[0]
    rest = shape[1:]
    return [_build_filler(rest, val) for _ in range(n)]

def _deepcopy_list(x):
    if isinstance(x, list):
        return [_deepcopy_list(e) for e in x]
    return x

# Linear algebra (minilinalg)
def dot(a: Array, b: Array) -> Number:
    if a.ndim == 1 and b.ndim == 1:
        if a.size != b.size:
            raise ValueError("dot requires vectors of same length")
        return sum(x * y for x, y in zip(a.data, b.data))
    elif a.ndim == 2 and b.ndim == 2:
        return matmul(a, b)
    else:
        raise NotImplementedError("dot supports 1D vectors or 2D matrices in this mini implementation")

def matmul(a: Array, b: Array) -> Array:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul requires 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError("shapes not aligned for matmul")
    m, k = a.shape
    _, n = b.shape
    out = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0
            for t in range(k):
                s += a.data[i][t] * b.data[t][j]
            out[i][j] = s
    return Array(out)

def norm(a: Array) -> float:
    flat = a._flatten()
    return math.sqrt(sum(x * x for x in flat))

def det(a: Array) -> Number:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("det requires a square 2D array")
    
    n = a.shape[0]
    temp_data = [row[:] for row in a.data] 
    sign = 1 

    for i in range(n):
        pivot_row = i
        for k in range(i + 1, n):
            if abs(temp_data[k][i]) > abs(temp_data[pivot_row][i]):
                pivot_row = k
        
        if abs(temp_data[pivot_row][i]) < 1e-12:
            return 0
            
        if pivot_row != i:
            temp_data[i], temp_data[pivot_row] = temp_data[pivot_row], temp_data[i]
            sign *= -1 

        for j in range(i + 1, n):
            if temp_data[j][i] == 0:
                continue
            
            factor = temp_data[j][i] / temp_data[i][i]
            for k in range(i, n):
                temp_data[j][k] -= factor * temp_data[i][k]
    result = 1
    for i in range(n):
        result *= temp_data[i][i]
        
    return result * sign

def inv(a: Array) -> Array:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("inv requires a square 2D array")
    
    n = a.shape[0]
    aug_data = []
    for i in range(n):
        row = a.data[i][:] + [1 if i == j else 0 for j in range(n)]
        aug_data.append(row)
    for i in range(n):
        pivot_row = i
        for k in range(i + 1, n):
            if abs(aug_data[k][i]) > abs(aug_data[pivot_row][i]):
                pivot_row = k
        if abs(aug_data[pivot_row][i]) < 1e-12:
            raise ValueError("Matrix is singular (det=0), cannot invert.")
        if pivot_row != i:
            aug_data[i], aug_data[pivot_row] = aug_data[pivot_row], aug_data[i]            

        pivot = aug_data[i][i]
        for j in range(i, 2 * n): 
            aug_data[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = aug_data[k][i]
                for j in range(i, 2 * n): 
                    aug_data[k][j] -= factor * aug_data[i][j]
                    
    inverse_data = [row[n:] for row in aug_data]
    return Array(inverse_data)
