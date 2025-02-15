# Numbersense-AI

This GitHub repository contains Python scripts and data (stimuli) that we used to investigate the visual number sense capabilities of some Foundation Models. This repo contains 2 main parts: Data collection and Benchmark. The data collection illustrates how the data can be collected from foundation models. The benchmark is used to evaluate the visual number sense capabilities of foundation models in different tasks.

## Data collection
The Python notebooks are used for generating the responses for foundation models in both numerosity naming task and numerosity production task.

All the data can be found in the [drive](https://unipdit-my.sharepoint.com/:f:/g/personal/kuinan_hou_studenti_unipd_it/Eod-B5Lc2yJErUMCQkQBwZ0BtI9JYKyDToD7CIh1hKebZg?e=YRiil3). The images from numerosity production task can be then processed by the following benchmark for evaluation. As for the numerosity naming task, users need to process the responses to create the confusion matrix and then use the evaluation script to get the metrics.
## Benchmark
Our benchmark goes beyond simple accuracy by incorporating the mean weighted error (MWE), 
a sophisticated metric that evaluates the precision of numerical reasoning; 
Counting-level, a concept derived from developmental psychology that mirrors stages of numerical understanding in humans; 
Similarity to human numerical estimation without counting: this metric measures how closely these agents emulate human number sense.

### Quick Start of the benchmark
The demo.ipynb notebook showcases how to run the benchmark with stable diffusion 2.1 on the fly (i.e. without saving the images locally) 

 Replace the current generate_image() fucntion with any API calls or multi-modal AI agents that return: 
- Image path
- Torch Tensor
- Numpy Array
- PIL image object

The final evaluation metrics will be saved in a dataframe.
### How to use the benchmark
In this section, we provide a detailed walk through of our benchmark, including how to use each module and how to interpret the results from our benchmark.
### 1. Installation
Clone Grounding DINO
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```
Make sure you have set CUDA_HOME as your environment variable. \
On Linux:
```bash
export CUDA_HOME=/path/to/cuda
```
On Windows:
```cmd
setx CUDA_HOME=/path/to/cuda`
```
**NB**: only complete version of CUDA will work, simply install cudatoolkit or nvcc from conda or other source is not possible \
Now install by the following commands: 
```bash
cd GroundingDINO/
pip install -e .
```
### 2. Prompt Generation with __generate_prompt.py__
This script generates a list of strings in the format:  
`"An image with {n} {object}"`  
where `n` ranges from 1 to 10, and the objects are chosen randomly from a predefined list of items.
#### Features
- Randomly selects singular or plural forms of objects based on the value of `n`.
- Supports saving the generated strings in `JSON`, `Pickle (PKL)`, or `Text (TXT)` formats.
- Configurable via command-line arguments.

#### Usage CLI
Use the command below to run the script:

```bash
python3 string_generator.py [options]
```
| **Option**        | **Description**                                     | **Default**       |
|---------------------------|-----------------------------------------------------|-------------------|
| `--num-strings`           | Number of strings to generate.                      | `100`             |
| `--format`         | Save format: `json`, `pkl`, or `txt`.               | None              |
| `--filename`       | Name of the output file (optional).                 | Auto-generated based on format. |

#### Usage in script
Import it as a function which will return the prompts in a python list.
```python
import generate_prompt
prompts = generate_prompt(save_format = None)
```

### 3. Evaluation
The evaluation can be done by first passing a image tensor/PIL Image object/image path/numpy array to eval_image() function which returns the generated numerosity. You should then create the confusion matrix as we explained in the demo. Finally, passing the confusion matrix to eval_CM() function that returns a panda dataframe with all the metrics.