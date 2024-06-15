from typing import List


def max_sliding_window(num_list: List[int], windows_size: int) -> List[int]:
    result = []
    window = num_list[:windows_size - 1]

    for index in range(windows_size - 1, len(num_list)):
        window.pop(0)
        window.append(num_list[index])
        result.append(max(window))

    return result


num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
windows_size = 3
print(max_sliding_window(num_list, windows_size))
