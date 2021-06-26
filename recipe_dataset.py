import json
import re
import random
import numpy as np
import mmh3
from typing import List



class DataIterator:
    def __init__(self, sentences: list) -> None:
        self._sentences = sentences
        self._offset = 0

    def reset(self, shuffle: bool=True) -> None:
        if shuffle:
            random.shuffle(self._sentences)
        else:
            # model will be faster with sentences sorted by size
            self._sentences = sorted(self._sentences, key=len)
        self._offset = 0

    def get(self, n: int) -> list:
        next_offset = min(len(self._sentences), self._offset + n)
        result = self._sentences[self._offset:next_offset]
        self._offset = next_offset
        return result

    def empty(self) -> bool:
        return self._offset >= len(self._sentences)


class Dataset:
    def __init__(self, json_files: list) -> None:
        """ json from
        https://eightportions.com/datasets/Recipes/#fn:1
        """
        assert len(json_files) > 0, "there must be at least one json file provided"
        numbers = re.compile("[0-9]+")
        spaces = re.compile("[ ]+")
        punct = re.compile("[,;&()]")
        self._sentences = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                content = json.load(f)
                for recipe in content.values():
                    instructions = recipe.get("instructions")
                    if instructions:
                        for s in instructions.lower().split("."):
                            s = numbers.sub(" N ", s)
                            s = punct.sub(" ", s)
                            s = s.strip()
                            if s:
                                words = np.array([
                                    mmh3.hash(w) for w in spaces.split(s)
                                ], dtype=np.uint32)
                                self._sentences.append(words)

    def split(self, *ratios) -> List[DataIterator]:
        random.shuffle(self._sentences)
        ratios = np.array(list(ratios))
        ratios /= np.sum(ratios)
        cutoff = ratios.cumsum()
        left = 0
        results = []
        for cut in cutoff:
            right = int(len(self._sentences) * cut)
            results.append(DataIterator(self._sentences[left:right]))
            left = right

        return results


if __name__ == "__main__":
    import sys
    dataset, _ = Dataset(sys.argv[1:]).split(0.5, 0.5)
    dataset.reset()
    print(dataset.get(10))
