import itertools

combinations = list(itertools.product([0, 1], repeat=5))
print(len(combinations))
