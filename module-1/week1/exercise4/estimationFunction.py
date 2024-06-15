import math


def factorial(n: int) -> int:
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def sin_approx(x: float, n: int) -> float:
    result = 0
    for i in range(0, n):
        term = ((-1) ** i) * (x ** (2 * i + 1)) / factorial(2 * i + 1)
        result += term
    return result


def cos_approx(x: float, n: int) -> float:
    result = 0
    for i in range(0, n):
        term = ((-1) ** i) * (x ** (2 * i)) / factorial(2 * i)
        result += term
    return result


def sinh_approx(x: float, n: int) -> float:
    result = 0
    for i in range(0, n):
        term = (x ** (2 * i + 1)) / factorial(2 * i + 1)
        result += term
    return result


def cosh_approx(x: float, n: int) -> float:
    result = 0
    for i in range(n):
        term = (x ** (2 * i)) / factorial(2 * i)
        result += term
    return result


def exercise4() -> None:
    x = float(input("Enter the value of x in radians: "))
    n = int(
        input("Enter the number of terms for the approximation (positive integer): "))

    if n <= 0:
        print("The number of terms must be a positive integer.")
    else:
        print(f"sin({x}) ≈ {sin_approx(x, n)}")
        print(f"cos({x}) ≈ {cos_approx(x, n)}")
        print(f"sinh({x}) ≈ {sinh_approx(x, n)}")
        print(f"cosh({x}) ≈ {cosh_approx(x, n)}")


exercise4()
