import scipy.spatial.distance as ds
import numpy as np
import timeit
from collections import deque



def compare_matrices(livemotion, usermotion):
    match = bool
    a = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 1, 1],
                  [1, 1, 0, 1, 0]])
    b = np.array([[1, 1, .8, 0, .6],
                  [.24, 0, 1.4, 2, 0],
                  [1, 1, 0, 1, 0]])

    ax = np.linalg.norm(a)
    bx = np.linalg.norm(b)
    set = """
import numpy as np
import scipy.spatial.distance as ds
    
    """
    first = """
def example():
    a = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 1, 1],
                  [1, 1, 0, 1, 0]])
    b = np.array([[1, 1, .8, 0, .6],
                  [.24, 0, 1.4, 2, 0],
                  [1, 1, 0, 1, 0]])
    np.sum(np.sum(np.square(a-b)), axis=1)
    """
    second = """
def example2():
    a = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 1, 1],
                  [1, 1, 0, 1, 0]])
    b = np.array([[1, 1, .8, 0, .6],
                  [.24, 0, 1.4, 2, 0],
                  [1, 1, 0, 1, 0]])
    similarity = 1 - ds.cdist(a, b, 'cosine')
    """
    print(timeit.timeit(setup=set, stmt=first, number=1000))

    print(timeit.timeit(setup=set, stmt=second, number=1000))
    # similarity = 1 - ds.cdist(a, b, 'cosine')
    #print(np.sum(similarity))

    return match


compare_matrices(livemotion=1, usermotion=1)
# queues
queue = deque([])
print(queue)
queue.append('the')
print(queue)
queue.append('bird')
print(queue)
queue.append('is')
print(queue)
queue.append('the')
print(queue)
queue.append('word')
print(queue)
# pops first element pushed in
queue.popleft()
print(queue)
# pops last element added 'stackwise'
queue.pop()
print(queue)