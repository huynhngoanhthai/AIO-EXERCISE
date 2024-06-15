
def precision(tp: int, fp: int) -> float:
    return tp/(tp + fp)


def recall(tp: int, fn: int) -> float:
    return tp/(tp + fn)


def f1_score(precision: float, recall: float) -> float:
    return 2 * (precision * recall)/(precision + recall)


def check_input_error(fn: int, tp: int, fp: int) -> None:
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
        check_input_error(fn, tp, fp)
        precision_value = precision(tp, fp)
        recall_value = recall(tp, fn)
        f1_score_value = f1_score(precision_value, recall_value)
        print('precision is', precision_value)
        print('recall is', recall_value)
        print('F1-score is', f1_score_value)
    except ValueError as e:
        print(e)


exercise1(tp=2, fp=4, fn=5)
