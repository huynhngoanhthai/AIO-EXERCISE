import math


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    return (x + math.sqrt(x**2)) / 2


def elu(x: float) -> float:
    alpha = 0.01
    return x if x > 0 else alpha * (math.exp(x) - 1)


def is_number(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def exercise2() -> None:
    x_input = input("Input x = ")
    if not is_number(x_input):
        print("x must be a number")
        return
    x = float(x_input)

    activation_function = input(
        "Input activation Function (sigmoid | relu | elu) : ").strip().lower()

    if activation_function == "sigmoid":
        result = sigmoid(x)
        print(f"sigmoid : f({x}) = {result}")
    elif activation_function == "relu":
        result = relu(x)
        print(f"relu : f({x}) = {result}")
    elif activation_function == "elu":
        result = elu(x)
        print(f"elu : f({x}) = {result}")
    else:
        print(f"{activation_function} is not supported")


exercise2()
