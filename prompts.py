prompt_1 = """
Here is a C code snippet:
{code}

Apply code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion to optimize this code for HLS. Provide the transformed code using markdown code block syntax and explain the rationale behind each transformation.
"""

prompt_1_1 = """
Here is a C code snippet:
{code}

Apply code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion to optimize this code for HLS. Provide the full transformed code (including all of the original pragmas) using markdown code block syntax and explain the rationale behind each transformation.
"""
