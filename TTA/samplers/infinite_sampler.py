import random
from typing import Sized


class InfiniteSampler:
    def __init__(self, dataset: Sized) -> None:
        r"""Randomly sample indexes of a dataset without replacement. Execute
        this process infinitely.
        """

        # Initialize indexes and pointer
        self.indexes = list(range(len(dataset)))
        # Shuffle indexes for the first time immediately after initialization
        random.shuffle(self.indexes)  # self.indexes: [3, 7, 0, ...]
        self.p = 0  # pointer

    # loop forever for sampling indexes when training
    def __iter__(self) -> int:
        r"""Yield an index."""

        while True:

            if self.p == len(self.indexes):
                
                # Traversed all data. Reshuffle indexes. Reset pointer.
                random.shuffle(self.indexes)
                self.p = 0
                
            index = self.indexes[self.p]
            self.p += 1

            yield index