
def precision(tp: int, fp: int) -> float:
    return tp/(tp + fp)


def recall(tp: int, fn: int) -> float:
    return tp/(tp + fn)


def f1Score(precision: float, recall: float) -> float:
    return 2 * (precision * recall)/(precision + recall)


def checkInputError(fn: int, tp: int, fp: int) -> None:
    if not isinstance(tp, int):
        raise ValueError("tp must be integers")
    if not isinstance(fn, int):
        raise ValueError("fn must be integers")
    if not isinstance(fp, int):
        raise ValueError("fp must be integers")
    if fn <= 0 or tp <= 0 or fp <= 0:
        raise ValueError("tp, fp, and fn must be greater than or equal to 0")


def exercise1(fn, tp, fp) -> None:
    try:
        checkInputError(fn, tp, fp)
        precisionValue = precision(tp, fp)
        recallValue = recall(tp, fn)
        f1ScoreValue = f1Score(precisionValue, recallValue)
        print('precision is', precisionValue)
        print('recall is', recallValue)
        print('F1-score is', f1ScoreValue)
    except ValueError as e:
        print(e)


# exercise1(tp=0, fp=3, fn=4)
# exercise1(tp=2, fp=3, fn=4)
# exercise1(tp=9, fp=3, fn=4)
# exercise1(tp=0, fp=0, fn=0)
# exercise1(tp=2.1, fp=3, fn=0)
exercise1(tp=2, fp=4, fn=5)
