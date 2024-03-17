#!/bin/bash

for file in *.c; do
    # Extract the base filename without the .c extension
    base_name=$(basename "$file" .c)

    echo "$file"

    # Compile the .c file into an executable, appending -bin to the base name
    gcc -o "${base_name}-bin" "$file" -lm
    if [ $? -eq 0 ]; then
        # echo "Compilation successful. Running ${base_name}-bin"
        # Execute the compiled program
        ./"${base_name}-bin"
    else
        echo "Compilation failed for $file."
    fi
done