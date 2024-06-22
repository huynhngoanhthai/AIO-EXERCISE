class Queue:
    __queue = []

    def __init__(self, capacity: int) -> None:
        self.__capacity = capacity

    def is_empty(self) -> bool:
        return len(self.__queue) == 0

    def is_full(self) -> bool:
        return len(self.__queue) == self.__capacity

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.__queue.pop(0)

    def enqueue(self, value) -> None:
        if self.is_full():
            raise OverflowError("enqueue to full queue")
        self.__queue.append(value)

    def front(self):
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self.__queue[0]


try:
    queue = Queue(3)
    print(queue.is_empty())
    queue.enqueue(1)
    queue.enqueue(2)
    print(queue.front())
    queue.enqueue(3)
    print(queue.is_full())
    queue.enqueue(4)
except OverflowError as e:
    print(e)
