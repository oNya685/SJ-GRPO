import ast

with open('recipe/sj-grpo/tests/raw_output.txt', 'r') as f:
    raw_output_list = ast.literal_eval(f.read())

import pysbd

for output in raw_output_list:
    print(output)
    print('===============')
    seg = pysbd.Segmenter(language="en", clean=False)
    print(seg.segment(output))
    print('===============')
    # please enter to get next output
    input()
