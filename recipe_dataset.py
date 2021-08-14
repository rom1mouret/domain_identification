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

    def reset(self, shuffle=True) -> None:
        if shuffle:
            random.shuffle(self._sentences)
        else:
            # model will run faster with sentences sorted by size
            self._sentences.sort(key=len)
        self._offset = 0

    def get(self, n: int, min_len: int=1) -> list:
        if min_len == 1:
            next_offset = min(len(self._sentences), self._offset + n)
            result = self._sentences[self._offset:next_offset]
            self._offset = next_offset
        else:
            result = []
            while len(result) < n and self._offset < len(self._sentences):
                candidate = self._sentences[self._offset]
                if len(candidate) >= min_len:
                    result.append(candidate)
                self._offset += 1
        return result

    def empty(self) -> bool:
        return self._offset >= len(self._sentences)


class Dataset:
    def __init__(self, json_files: list, min_len: int=1, rm_duplicates: bool=False) -> None:
        """ json from
        https://eightportions.com/datasets/Recipes/#fn:1
        """
        assert len(json_files) > 0, "there must be at least one json file provided"
        numbers = re.compile("[0-9]+")
        spaces = re.compile("[ ]+")
        punct = re.compile("[,;&()]")
        sentences = []
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
                                words = tuple([
                                    mmh3.hash(w) for w in spaces.split(s)
                                ])
                                if len(words) >= min_len:
                                    sentences.append(words)
        if rm_duplicates:
            size_before = len(sentences)
            sentences = list(set(sentences))
            size_after = len(sentences)
            print("duplicates removed:", size_before - size_after, "/", size_before)

        self._sentences = [np.array(words, dtype=np.uint32) for words in sentences]
        print("number of sentences:", len(self._sentences))

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
