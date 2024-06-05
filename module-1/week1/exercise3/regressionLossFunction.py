import random
import math


def mae(predictions: list[float], targets: list[float]) -> float:
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)


def mse(predictions: list[float], targets: list[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def rmse(predictions: list[float], targets: list[float]) -> float:
    return math.sqrt(mse(predictions, targets))


def exercise3() -> None:
    num_samples = input(
        "Input number of samples ( integer number ) which are generated: ")
    if not num_samples.isnumeric():
        print("number of samples must be an integer number")
        return

    num_samples = int(num_samples)

    loss_name = input("Enter loss name (MAE, MSE, RMSE): ").strip().upper()

    predictions = []
    targets = []

    for i in range(num_samples):
        predict = random.uniform(0, 10)
        target = random.uniform(0, 10)
        predictions.append(predict)
        targets.append(target)
        print(f"sample-{i} | predict: {predict:.4f} | target: {target:.4f}")

    if loss_name == "MAE":
        loss = mae(predictions, targets)
    elif loss_name == "MSE":
        loss = mse(predictions, targets)
    elif loss_name == "RMSE":
        loss = rmse(predictions, targets)

    print(f"Loss name: {loss_name}")
    print(f"Loss: {loss:.4f}")


exercise3()
