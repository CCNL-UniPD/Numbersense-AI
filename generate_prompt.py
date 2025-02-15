"""
String Generator Script

This script generates a list of strings in the format "{n} {object}" where `n` 
ranges from 1 to 10, and the objects are chosen randomly from a predefined list.

Usage:
    - Run the script directly and specify the number of strings, format, and filename as needed.
    - Outputs a list of generated strings to stdout and optionally saves them in the specified format.

Author: Kuinan Hou
License: CC0 1.0 Universal
"""

import json
import pickle
import argparse
import random

def generate_strings(num_strings=100, save_format='txt', filename=None):
    """
    Generate a list of strings of format: "An image with {n} {object}", where
    n ranges from 1 to 10 and objects are chosen randomly from a predefined list.

    Each number (1-10) is equally represented, and categories are randomized.

    Parameters:
        num_strings (int): Number of strings to generate.
        save_format (str): One of 'json', 'pkl', or 'txt', or None.
        filename (str): Filename to save the output to. If None, a default name is chosen.

    Returns:
        list: The generated list of strings.
    """
    # Define a dictionary mapping plural to singular forms
    objects_map = {
        "apples": "apple",
        "people": "person",
        "human faces": "human face",
        "dots": "dot",
        "butterflies": "butterfly",
        "triangles": "triangle",
        "dogs": "dog",
        "cars": "car",
        "pencils": "pencil",
        "shoes": "shoe",
        "guitars": "guitar"
    }

    plural_objects = list(objects_map.keys())
    generated = []

    # Ensure equal sampling of numbers from 1 to 10
    numbers = list(range(1, 11)) * (num_strings // 10) + list(range(1, (num_strings % 10) + 1))
    random.shuffle(numbers)  # Shuffle the numbers

    for n in numbers:
        # Pick a random object from the list
        obj_plural = random.choice(plural_objects)
        obj_singular = objects_map[obj_plural]

        # Decide whether to use singular or plural based on n
        if n == 1:
            text = f"An image with {n} {obj_singular}"
        else:
            text = f"An image with {n} {obj_plural}"

        generated.append(text)

    # If a save format is specified, save the generated list
    if save_format is not None:
        if filename is None:
            filename = f"prompts.{save_format}"

        if save_format == 'json':
            with open(filename, "w") as json_file:
                json.dump(generated, json_file, indent=4)
        elif save_format == 'pkl':
            with open(filename, "wb") as pkl_file:
                pickle.dump(generated, pkl_file)
        elif save_format == 'txt':
            with open(filename, "w") as txt_file:
                txt_file.write(",".join(generated))
        else:
            raise ValueError("Unsupported format. Please choose from 'json', 'pkl', or 'txt'.")

    return generated

def main():
    parser = argparse.ArgumentParser(description="Generate a list of strings and optionally save them.")
    parser.add_argument("--num-strings", type=int, default=100, help="Number of strings to generate.")
    parser.add_argument("--format", type=str, choices=["json", "pkl", "txt"], default='txt', help="Save format.")
    parser.add_argument("--filename", type=str, default=None, help="Output filename (optional).")
    args = parser.parse_args()

    generate_strings(num_strings=args.num_strings, save_format=args.format, filename=args.filename)
    print(f'Results saved as {args.format}')

if __name__ == "__main__":
    main()