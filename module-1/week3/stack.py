class Stack:
    __stack = []

    def __init__(self, capacity: int) -> None:
        self.__capacity = capacity

    def is_empty(self) -> bool:
        return len(self.__stack) == 0

    def is_full(self) -> bool:
        return len(self.__stack) == self.__capacity

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.__stack.pop()

    def push(self, value) -> None:
        if self.is_full():
            raise OverflowError("push to full stack")
        self.__stack.append(value)

    def top(self):
        if self.is_empty():
            raise IndexError("top from empty stack")
        return self.__stack[-1]


try:
    stack = Stack(capacity=3)
    stack.push(1)
    stack.push(2)
    print(stack.top())
except OverflowError as e:
    print(e)
