import subprocess
import re


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def write_to_file(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)


def compile_and_run(source_file, executable_name="test_program"):
    compile_command = ["gcc", source_file, "-o", executable_name, "-lm"]
    try:
        subprocess.run(compile_command, check=True)
        result = subprocess.run(f"./{executable_name}", capture_output=True, text=True)
        subprocess.run(["rm", "-f", executable_name])
        return result.stdout
    except subprocess.CalledProcessError as e:
        return str(e.stderr) + "\n"


def extract_and_rename_functions(code, new_function_name):
    # Replace any function name with new_function_name
    modified_code = re.sub(r"void\s+(\w+)\s*\(", f"void {new_function_name}(", code)

    return modified_code


def main():
    selected_kernels = [
        "bicg_kernel.c",
        "doitgen_kernel.c",
        "atax_kernel.c",
        "gemver_kernel.c",
        "syrk_kernel.c",
        "md_kernel.c",
        "heat-3d_kernel.c",
        "fdtd-2d_kernel.c",
        "stencil_stencil2d_kernel.c",
        "adi_kernel.c",
        "seidel-2d_kernel.c",
        "covariance_kernel.c",
        "correlation_kernel.c",
    ]

    test_kernels = [
        "test_bicg.c",
        "test_doitgen.c",
        "test_atax.c",
        "test_gemver.c",
        "test_syrk.c",
        "test_md.c",
        "test_heat-3d.c",
        "test_fdtd-2d.c",
        "test_stencil_stencil2d.c",
        "test_adi.c",
        "test_seidel-2d.c",
        "test_covariance.c",
        "test_correlation.c",
    ]

    # TODO
    models = ["gpt-3.5-turbo", "gpt-4-turbo-preview"]
    prompts = []
    res = []
    result = []

    for i in range(1, 6):

        original_folder = "../data/selected_sources/"
        transformed_folder = (
            f"../transformed_sources/gpt-4-turbo-preview/prompt_3_res_{i}/"
        )

        for code_file, test_file in zip(selected_kernels, test_kernels):

            # File paths
            original_kernel_path = original_folder + code_file
            transformed_kernel_path = transformed_folder + code_file
            test_file_path = test_file

            # Read the function codes
            kernel_original_code = read_file(original_kernel_path)
            kernel_transformed_code = read_file(transformed_kernel_path)
            test_code = read_file(test_file_path)

            main_func = re.search(r"int\s+main\s*\(.*?\)\s*{.*}", test_code, re.DOTALL)

            # Extract the body of functions if they are found
            init_array_code = """
void init_array(double *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
    }
}
"""
            compare_arrays_code = """
int compare_arrays(double *arr1, double *arr2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(arr1[i] - arr2[i]) > 1e-6)
        {             // Using a tolerance to account for floating-point arithmetic differences
            return 0; // Arrays are not the same
        }
    }
    return 1; // Arrays are the same
}
"""
            main_code = main_func.group(0) if main_func else ""

            kernel_original_code = extract_and_rename_functions(
                kernel_original_code,
                "kernel" + test_file[4:-2].replace("-", "_") + "_original",
            )

            kernel_transformed_code = extract_and_rename_functions(
                kernel_transformed_code,
                "kernel" + test_file[4:-2].replace("-", "_") + "_transformed",
            )

            # Combine codes
            combined_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
{kernel_original_code}
{kernel_transformed_code}
{init_array_code}
{compare_arrays_code}
{main_code}
    """

            # Write the combined code to a new C file
            combined_file_path = test_file
            write_to_file(combined_file_path, combined_code)

            # Compile and run
            output = compile_and_run(combined_file_path)

            # Store result
            result.append(transformed_kernel_path[3:] + ": " + output)

    with open("results_gpt-4-turbo-preview_prompt_2.txt", "a") as f:
        for resul in result:
            f.write(resul)
        f.close()


if __name__ == "__main__":
    main()
