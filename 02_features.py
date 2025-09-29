import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import gurobipy as gp
from utils import features, compute_epslabels

parser = argparse.ArgumentParser()
parser.add_argument("-problem", type=int, default=0)
args = parser.parse_args()

base_dir = "./goodinstances/"
all_instances = os.listdir(base_dir)
for file in all_instances:
    if ".csv" in file:
        all_instances.remove(file)
all_instances.sort()


problem_name = all_instances[args.problem].split(sep=".")[0]

dir_name = "./goodfiles/" + problem_name + "/results/"

files = [file for file in os.listdir(dir_name) if "problem" not in file]
files.sort()

print(len(files))

df_list = []
col_names = [
    "rounds",
    "lp_time",
    "sep_status",
    "sep_time",
    "num_cuts",
    "gap_closed",
    "sep_gap",
]
for file in files:
    data = pd.read_csv(
        dir_name + file,
        header=None,
        names=col_names,
    )
    df_list.append(data)

df_all = pd.concat([d.iloc[-1:] for d in df_list], ignore_index=True)

bad_files = df_all[df_all["gap_closed"] > 100]
# how bad is this code?
print("bad files: ", len(bad_files))

good_files = df_all[df_all["gap_closed"] <= 100]
# good_files = df_all[df_all["gap_closed"] >= 5]
# print(len(good_files))


good_indx = good_files.index.tolist()
print("good files: ", len(good_indx))
# print(good_indx)

# boxplot of gap_closed and save to file
box = sns.boxplot(x=good_files["gap_closed"])
fig = box.get_figure()
fig.savefig(problem_name + ".png")

# my_idx = good_idx[args.index]

directory = (
    # "/blue/akazachkov/o.guaje/" + "gooddata/" + problem_name + "/"
    "./goodfiles/"
    + problem_name
    + "/"
)

data_dir = directory + "data/"
try:
    os.makedirs(data_dir, exist_ok=True)
except FileExistsError:
    pass

# good_indx.remove(810)
for my_idx in good_indx:

    ya_existe = os.path.isfile(data_dir + str(my_idx).zfill(4) + ".csv")

    if not ya_existe:
        read_dir = directory + "random/"

        alphas_dir = directory + "cuts/" + str(my_idx).zfill(4) + "/"

        sols_dir = directory + "sols/" + str(my_idx).zfill(4) + "/"

        multipliers_dir = directory + "lambdas/" + str(my_idx).zfill(4) + "/"

        ip = gp.read(read_dir + str(my_idx).zfill(4) + ".mps")
        var_types = ip.getAttr("VType")
        bounds = ip.getAttr("UB")

        for i in range(len(bounds)):
            if bounds[i] == 1 and var_types[i] == "I":
                var_types[i] = "B"

        num_rounds = len(os.listdir(sols_dir))

        lp = ip.relax()
        lp.optimize()

        solution = np.array([v.x for v in lp.getVars()])

        all_data = []

        for i in range(num_rounds):
            this_dataset, feature_names = features(lp, solution, var_types)
            this_dataset = pd.DataFrame(this_dataset, columns=feature_names)
            labels = np.loadtxt(
                multipliers_dir + str(i) + ".csv", delimiter=","
            )
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
            iter_label = [i] * this_dataset.shape[0]
            instance_label = [my_idx] * this_dataset.shape[0]
            label_1 = [
                1 if len(np.nonzero(labels[:, i])[0]) else 0
                for i in range(labels.shape[1])
            ]
            label_2 = compute_epslabels(labels, 0.1)
            label_3 = np.max(np.abs(labels), axis=0)
            this_dataset["instance_id"] = instance_label
            this_dataset["cut_iter"] = iter_label
            this_dataset["label_any"] = label_1
            this_dataset["label_eps"] = label_2
            this_dataset["label_max"] = label_3
            all_data.append(this_dataset)
            solution = np.loadtxt(sols_dir + str(i) + ".csv", delimiter=",")

        huge_dataset = pd.concat(all_data)
        huge_dataset["constraint_id"] = pd.to_numeric(
            huge_dataset["constraint_id"], downcast="unsigned"
        )
        huge_dataset["slack"] = pd.to_numeric(
            huge_dataset["slack"], downcast="float"
        )
        huge_dataset["is_active"] = huge_dataset["is_active"].astype(
            "category"
        )
        huge_dataset["dual"] = pd.to_numeric(
            huge_dataset["dual"], downcast="float"
        )
        huge_dataset["sense_eq"] = huge_dataset["sense_eq"].astype("category")
        huge_dataset["sense_leq"] = huge_dataset["sense_leq"].astype(
            "category"
        )
        huge_dataset["degree"] = pd.to_numeric(
            huge_dataset["degree"], downcast="unsigned"
        )
        huge_dataset["zero_rhs"] = huge_dataset["zero_rhs"].astype("category")
        huge_dataset["mean_coefficients"] = pd.to_numeric(
            huge_dataset["mean_coefficients"], downcast="float"
        )
        huge_dataset["std_coefficients"] = pd.to_numeric(
            huge_dataset["std_coefficients"], downcast="float"
        )
        huge_dataset["min_coeffs"] = pd.to_numeric(
            huge_dataset["min_coeffs"], downcast="float"
        )
        huge_dataset["max_coeffs"] = pd.to_numeric(
            huge_dataset["max_coeffs"], downcast="float"
        )
        huge_dataset["max_ratio"] = pd.to_numeric(
            huge_dataset["max_ratio"], downcast="float"
        )
        huge_dataset["min_ratio"] = pd.to_numeric(
            huge_dataset["min_ratio"], downcast="float"
        )
        huge_dataset["mean_ratio"] = pd.to_numeric(
            huge_dataset["mean_ratio"], downcast="float"
        )
        huge_dataset["std_ratio"] = pd.to_numeric(
            huge_dataset["std_ratio"], downcast="float"
        )
        huge_dataset["mean_coefficients_non0"] = pd.to_numeric(
            huge_dataset["mean_coefficients_non0"], downcast="float"
        )
        huge_dataset["std_coefficients_non0"] = pd.to_numeric(
            huge_dataset["std_coefficients_non0"], downcast="float"
        )
        huge_dataset["min_coeffs_non0"] = pd.to_numeric(
            huge_dataset["min_coeffs_non0"], downcast="float"
        )
        huge_dataset["max_coeffs_non0"] = pd.to_numeric(
            huge_dataset["max_coeffs_non0"], downcast="float"
        )
        huge_dataset["max_ratio_non0"] = pd.to_numeric(
            huge_dataset["max_ratio_non0"], downcast="float"
        )
        huge_dataset["min_ratio_non0"] = pd.to_numeric(
            huge_dataset["min_ratio_non0"], downcast="float"
        )
        huge_dataset["mean_ratio_non0"] = pd.to_numeric(
            huge_dataset["mean_ratio_non0"], downcast="float"
        )
        huge_dataset["std_ratio_non0"] = pd.to_numeric(
            huge_dataset["std_ratio_non0"], downcast="float"
        )
        huge_dataset["mean_coefficients_0"] = pd.to_numeric(
            huge_dataset["mean_coefficients_0"], downcast="float"
        )
        huge_dataset["std_coefficients_0"] = pd.to_numeric(
            huge_dataset["std_coefficients_0"], downcast="float"
        )
        huge_dataset["min_coeffs_0"] = pd.to_numeric(
            huge_dataset["min_coeffs_0"], downcast="float"
        )
        huge_dataset["max_coeffs_0"] = pd.to_numeric(
            huge_dataset["max_coeffs_0"], downcast="float"
        )
        huge_dataset["max_ratio_0"] = pd.to_numeric(
            huge_dataset["max_ratio_0"], downcast="float"
        )
        huge_dataset["min_ratio_0"] = pd.to_numeric(
            huge_dataset["min_ratio_0"], downcast="float"
        )
        huge_dataset["mean_ratio_0"] = pd.to_numeric(
            huge_dataset["mean_ratio_0"], downcast="float"
        )
        huge_dataset["std_ratio_0"] = pd.to_numeric(
            huge_dataset["std_ratio_0"], downcast="float"
        )
        huge_dataset["mean_coefficients_UB"] = pd.to_numeric(
            huge_dataset["mean_coefficients_UB"], downcast="float"
        )
        huge_dataset["std_coefficients_UB"] = pd.to_numeric(
            huge_dataset["std_coefficients_UB"], downcast="float"
        )
        huge_dataset["min_coeffs_UB"] = pd.to_numeric(
            huge_dataset["min_coeffs_UB"], downcast="float"
        )
        huge_dataset["max_coeffs_UB"] = pd.to_numeric(
            huge_dataset["max_coeffs_UB"], downcast="float"
        )
        huge_dataset["max_ratio_UB"] = pd.to_numeric(
            huge_dataset["max_ratio_UB"], downcast="float"
        )
        huge_dataset["min_ratio_UB"] = pd.to_numeric(
            huge_dataset["min_ratio_UB"], downcast="float"
        )
        huge_dataset["mean_ratio_UB"] = pd.to_numeric(
            huge_dataset["mean_ratio_UB"], downcast="float"
        )
        huge_dataset["std_ratio_UB"] = pd.to_numeric(
            huge_dataset["std_ratio_UB"], downcast="float"
        )
        huge_dataset["nonzero_coeffs_x"] = pd.to_numeric(
            huge_dataset["nonzero_coeffs_x"], downcast="unsigned"
        )
        huge_dataset["euclidean_distance"] = pd.to_numeric(
            huge_dataset["euclidean_distance"], downcast="float"
        )
        huge_dataset["relative_violation"] = pd.to_numeric(
            huge_dataset["relative_violation"], downcast="float"
        )
        huge_dataset["adjusted_distance"] = pd.to_numeric(
            huge_dataset["adjusted_distance"], downcast="float"
        )
        huge_dataset["cost_mean"] = pd.to_numeric(
            huge_dataset["cost_mean"], downcast="float"
        )
        huge_dataset["cost_min"] = pd.to_numeric(
            huge_dataset["cost_min"], downcast="float"
        )
        huge_dataset["cost_max"] = pd.to_numeric(
            huge_dataset["cost_max"], downcast="float"
        )
        huge_dataset["cost_std"] = pd.to_numeric(
            huge_dataset["cost_std"], downcast="float"
        )
        huge_dataset["num_in_1_costs"] = pd.to_numeric(
            huge_dataset["num_in_1_costs"], downcast="unsigned"
        )
        huge_dataset["num_in_5_costs"] = pd.to_numeric(
            huge_dataset["num_in_5_costs"], downcast="unsigned"
        )
        huge_dataset["num_in_10_costs"] = pd.to_numeric(
            huge_dataset["num_in_10_costs"], downcast="unsigned"
        )
        huge_dataset["num_in_20_costs"] = pd.to_numeric(
            huge_dataset["num_in_20_costs"], downcast="unsigned"
        )
        huge_dataset["frac_in_1_costs"] = pd.to_numeric(
            huge_dataset["frac_in_1_costs"], downcast="float"
        )
        huge_dataset["frac_in_5_costs"] = pd.to_numeric(
            huge_dataset["frac_in_5_costs"], downcast="float"
        )
        huge_dataset["frac_in_10_costs"] = pd.to_numeric(
            huge_dataset["frac_in_10_costs"], downcast="float"
        )
        huge_dataset["frac_in_20_costs"] = pd.to_numeric(
            huge_dataset["frac_in_20_costs"], downcast="float"
        )
        huge_dataset["is_singleton"] = huge_dataset["is_singleton"].astype(
            "category"
        )
        huge_dataset["is_aggregation"] = huge_dataset["is_aggregation"].astype(
            "category"
        )
        huge_dataset["is_precedence"] = huge_dataset["is_precedence"].astype(
            "category"
        )
        huge_dataset["is_varbound"] = huge_dataset["is_varbound"].astype(
            "category"
        )
        huge_dataset["is_setpartion"] = huge_dataset["is_setpartion"].astype(
            "category"
        )
        huge_dataset["is_setpacking"] = huge_dataset["is_setpacking"].astype(
            "category"
        )
        huge_dataset["is_setcovering"] = huge_dataset["is_setcovering"].astype(
            "category"
        )
        huge_dataset["is_cardinality"] = huge_dataset["is_cardinality"].astype(
            "category"
        )
        huge_dataset["is_invariantknapsack"] = huge_dataset[
            "is_invariantknapsack"
        ].astype("category")
        huge_dataset["is_equationknapsack"] = huge_dataset[
            "is_equationknapsack"
        ].astype("category")
        huge_dataset["is_binpacking"] = huge_dataset["is_binpacking"].astype(
            "category"
        )
        huge_dataset["is_knapsack"] = huge_dataset["is_knapsack"].astype(
            "category"
        )
        huge_dataset["is_integerknapsack"] = huge_dataset[
            "is_integerknapsack"
        ].astype("category")
        huge_dataset["is_mixedbinary"] = huge_dataset["is_mixedbinary"].astype(
            "category"
        )
        huge_dataset["instance_id"] = pd.to_numeric(
            huge_dataset["instance_id"], downcast="unsigned"
        )
        huge_dataset["cut_iter"] = pd.to_numeric(
            huge_dataset["cut_iter"], downcast="unsigned"
        )
        huge_dataset["label_any"] = huge_dataset["label_any"].astype(
            "category"
        )
        huge_dataset["label_eps"] = huge_dataset["label_eps"].astype(
            "category"
        )
        huge_dataset["label_max"] = pd.to_numeric(
            huge_dataset["label_max"], downcast="float"
        )

        print(huge_dataset.shape)
        huge_dataset.to_csv(
            data_dir + str(my_idx).zfill(4) + ".csv", index=False
        )
