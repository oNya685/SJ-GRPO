import re
from typing import Any
import importlib

from torch import chunk

sj_grpo = importlib.import_module('recipe.sj-grpo.segmenter')
Segmenter = sj_grpo.Segmenter

if __name__ == "__main__":

    # 测试
    import ast

    with open('recipe/sj-grpo/tests/raw_output.txt', 'r') as f:
        raw_output_list = ast.literal_eval(f.read())

    for output in raw_output_list:
        print(output)
        print('===============')
        segmenter = Segmenter()
        res = segmenter.segment(output)
        for i, s in enumerate(res):
            print(f"--- Chunk {i+1} ---")
            print(s)
        print('===============')
        # press enter to get next output
        input()
