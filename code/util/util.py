class Stack:
    stack = None

    def __init__(self):
        self.stack = []

    def push(self, element):
        self.stack.append(element)

    def pop(self):
        return self.stack.pop()

    def is_empty(self):
        if len(self.stack) is 0:
            return True

        return False


class Queue:
    queue = None

    def __init__(self):
        self.queue = []

    def push(self, element):
        self.queue.append(element)

    def pop(self):
        return self.queue.pop(0)

    def is_empty(self):
        if len(self.queue) is 0:
            return True

        return False


class PriorityQueue:
    priority_queue = None

    def __init__(self):
        self.priority_queue = []

    def push(self, element, heuristic):
        self.priority_queue.append((element, heuristic))
        self.priority_queue.sort(key=lambda tup: tup[1], reverse=True)

    def pop(self):
        return self.priority_queue.pop()

    def is_empty(self):
        if len(self.priority_queue) is 0:
            return True

        return False
