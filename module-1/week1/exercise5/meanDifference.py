import math


def md_nre_single(y: float, y_hat: float, n: int, p: int) -> float:
    return (y**(1/n) - y_hat**(1/n))**p


def MD_nRE(md_nre_list: list[float]) -> float:
    if not md_nre_list:
        return 0.0
    return sum(md_nre_list) / len(md_nre_list)


md_nre_values = [
    md_nre_single(10, 8, 2, 3),
    md_nre_single(15, 10, 2, 3),
    md_nre_single(7, 5, 2, 3)
]

print(MD_nRE(md_nre_values))
