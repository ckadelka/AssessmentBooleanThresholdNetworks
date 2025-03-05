# AssessmentBooleanThresholdNetworks

A pipeline to critically assess the ability of threshold networks to accurately describe the dynamics of Boolean gene regulatory networks

The main file, assess_threshold_rules.py, contains all code to perform the analyses described in the paper "Critical assessment of the ability of Boolean threshold models to describe gene regulatory network dynamics", available at [https://arxiv.org/abs/xxx](https://arxiv.org/abs/xxx).

The published expert-curated biological networks are copy pasted from [https://github.com/ckadelka/DesignPrinciplesGeneNetworks](https://github.com/ckadelka/DesignPrinciplesGeneNetworks). This repository also contains an older version of load_database.py and canalizing_function_toolbox.py, which are updated here.

# load_database.py
This program loads all published expert-curated Boolean network models from a recent [meta-analysis](https://www.science.org/doi/full/10.1126/sciadv.adj0822), whose nodes have a predefined maximal in-degree (14 used in the paper; to avoid an exponential increase in run time) The models are stored in a list of folders, as text files in a standardized format:
```text
A = B OR C
B = A OR (C AND D)
C = NOT A
```
This little example represents a model with three genes, A, B and C, and one external parameter D (which only appears on the right side of the equations).

# canalizing_function_toolbox.py
This file contains a variety of functions to analyze Boolean functions and Boolean networks. Each Python function has its own documentation. A Boolean function is considered as a list of 0-1 entries of length 2^n where n is the number of inputs. A Boolean network of N nodes is a list of N Boolean functions. For example,
```python
f_A = [0,1,1,1]
f_B = [0,0,0,1,1,1,1,1]
f_C = [1,0]
F = [f_A,f_B,f_C]
```
describes the Boolean network from above. One can also get this via
```python
import load_database as db

with open('example.txt', 'w') as writer:
    writer.write('A = B OR C\nB = A OR (C AND D)\nC = NOT A')
F, I, degree, variables, constants = db.text_to_BN(folder='',textfile='example.txt')
```
which yields in addition the adjacency matrix I, the in-degree of each node, the names of the variables (in order) and the names of potential external parameters.

