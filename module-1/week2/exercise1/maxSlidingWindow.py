from typing import List


def maxSlidingWindow(numList: List[int], windowsSize: int) -> List[int]:
    result = []
    window = numList[:windowsSize - 1]

    for index in range(windowsSize - 1, len(numList)):
        window.pop(0)
        window.append(numList[index])
        result.append(max(window))

    return result


numList = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
windowsSize = 3
print(maxSlidingWindow(numList, windowsSize))
