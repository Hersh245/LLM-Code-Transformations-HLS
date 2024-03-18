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

prompt_2 = """
Here is a C code snippet:
{code}

Apply code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion to optimize this code for HLS. Provide the full transformed code (including all of the original pragmas) using markdown code block syntax and explain the rationale behind each transformation. Please make sure that the transformed code has same functionality as the original code. 
"""

prompt_3 = """
Here is a C code snippet:

{code}

Using the provided performance estimate from the merlin.rpt file below, apply code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion to optimize this code for High-Level Synthesis (HLS). 
Consider the impact of each transformation on the trip count (TC), accumulated cycles (AC), and cycles per call (CPC) for improved performance. 
Ensure to include all of the original pragmas in the transformed code. Use markdown code block syntax to present the full transformed code and explain the rationale behind each transformation based on the performance estimates.

{merlin_table}
"""
