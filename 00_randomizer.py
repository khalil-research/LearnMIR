##############################################################################
# Learning to Cut. Oscar Guaje, Arnaud Deza, Aleksandr Kazachkov, Elias Khalil
# Script to randomize and IP instance.
##############################################################################

import os
import gurobipy as gp
import numpy as np
import argparse
from utils import is_integer, existing_sol

parser = argparse.ArgumentParser()
parser.add_argument("-problem", type=int, default=0)
parser.add_argument("-instances", type=int, default=10)
args = parser.parse_args()


directory = "./goodinstances/"
files = os.listdir(directory)

for file in files:
    if ".csv" in file:
        files.remove(file)

files.sort()

problem = files[args.problem].split(sep=".")[0]

max_instances = args.instances
problem_path = "./scratch/goodfiles/" + problem + "/"
instance = "./goodinstances/" + problem

# Read the instance
m = gp.read(instance + ".mps")

# Create the file structure for the output
instances_path = problem_path + "random/"
try:
    os.makedirs(problem_path)
except OSError as error:
    print(error)
try:
    os.makedirs(instances_path)
except OSError as error:
    print(error)


np.random.seed(153)

num_instances = 0
solutions = []
tries = 0

m_copy = m.copy()
m_copy.Params.OutputFlag = 0
m_copy.Params.LogFile = ""
m_copy.Params.Cuts = 0
m_copy.Params.Presolve = 0
m_copy.Params.Heuristics = 0

m_variables = m_copy.getVars()

xs = [i for i in m_variables if i.Obj < -1e-6]
ys = [i for i in m_variables if i.Obj > 1e-6]

x_mean = 0
x_sd = 0
if len(xs) > 0:
    x_mean = np.mean([i.Obj for i in xs])
    # x_sd = np.std([i.Obj for i in xs])
    x_sd = np.max(np.abs([i.Obj for i in xs]))

y_mean = 0
y_sd = 0
if len(ys) > 0:
    y_mean = np.mean([i.Obj for i in ys])
    # y_sd = np.std([i.Obj for i in ys])
    y_sd = np.max(np.abs([i.Obj for i in ys]))

while num_instances < max_instances and tries < 10000:

    if len(xs) > 0:
        for x in xs:
            x.Obj = min(np.random.normal(x_mean, x_sd), 0)
            # x.Obj = float(np.random.normal(x_mean, x_sd))

    if len(ys) > 0:
        for y in ys:
            y.Obj = max(np.random.normal(y_mean, y_sd), 0)
            # y.Obj = float(np.random.normal(y_mean, y_sd))

    ########################################################

    m_copy.update()

    m_relax = m_copy.relax()
    m_relax.optimize()

    current_sol = np.array([var.X for var in m_relax.getVars()])

    # print(existing_sol(solutions, current_sol), "solution exists")
    # print(not is_integer(current_sol), "solution is fractional")

    if not existing_sol(solutions, current_sol) and not is_integer(
        current_sol
    ):
        solutions.append(current_sol)
        filename = instances_path + str(num_instances).zfill(4) + ".mps"
        m_copy.write(filename)

        num_instances = num_instances + 1

    tries = tries + 1
    # m = m_copy.copy()

# with open(problem_path + "samples", "w") as f:
#     f.write(str(tries))

# with open("./randomizer_results", "a") as f:
#     f.write(problem + "\n")
