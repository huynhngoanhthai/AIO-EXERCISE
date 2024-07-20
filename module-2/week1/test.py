import numpy as np
arr = np.array([1, 2, 3])
# arr[arr % 2 == 1] = -1
# print(arr)
# a_2d = arr.reshape(2, -1)
# arr2 = np.repeat(1, 10).reshape(2, -1)

# c = np.concatenate([a_2d, arr2], axis=0)
print(np . repeat(arr, 3))
print(np . tile(arr, 3))

# print(c)
