from openai import OpenAI
from config import API_KEY
import os
import re

client = OpenAI(api_key=API_KEY)


def transform_code_with_gpt(file_path, prompt_template_1):
    with open(file_path, "r") as file:
        code = file.read()
    prompt = prompt_template_1.format(code=code)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Or the latest available version
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        n=1,
        stop=None,
    )
    return response.choices[0].message.content


def parse_and_save_transformed_code(
    transformed_text, original_file_name, target_folder
):
    # Regex to find code block between ```c and ```
    code_block_pattern = re.compile(r"```c\n([\s\S]*?)\n```")
    code_match = code_block_pattern.search(transformed_text)

    if code_match:
        # Extract code and the rest of the response
        transformed_code = code_match.group(1)
        before_code = transformed_text[: code_match.start()].strip()
        after_code = transformed_text[code_match.end() :].strip()

        # Combine comments from before and after the code, if they exist
        comments = ""
        if before_code:
            comments += (
                "\n".join(["// " + line for line in before_code.split("\n")]) + "\n"
            )
        comments += transformed_code
        if after_code:
            comments += "\n" + "\n".join(
                ["// " + line for line in after_code.split("\n")]
            )

        # Path for the new file
        target_file_path = os.path.join(target_folder, original_file_name)
        with open(target_file_path, "w") as file:
            file.write(comments)
        print(f"Transformed code with comments saved to {target_file_path}")
    else:
        print(f"Code block not found in the response for {original_file_name}")
        print(transformed_text)


# Example prompt template
prompt_template_1 = """
Here is a C code snippet:
{code}

Apply code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion to optimize this code for HLS. Provide the transformed code using markdown code block syntax and explain the rationale behind each transformation.
"""

# Directory to store transformed files
target_folder = "./transformed_sources_1"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Iterate over all files in data/sources
source_folder = "data/sources"

if __name__ == "__main__":
    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)
        if os.path.isfile(file_path):
            transformed_text = transform_code_with_gpt(file_path, prompt_template_1)
            parse_and_save_transformed_code(transformed_text, file_name, target_folder)
