import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures


# Functions to check if a solution is already added and if it is integer
def existing_sol(sols, new_sol):
    comparisons = [
        np.allclose(sols[i], new_sol, atol=1.0e-4) for i in range(len(sols))
    ]
    return np.sum(comparisons) > 0


def is_integer(new_sol):
    for i in range(len(new_sol)):
        if new_sol[i] - math.floor(new_sol[i]) > 1.0e-4 and new_sol[
            i
        ] - math.floor(new_sol[i]) < (1 - 1.0e-4):
            return False
    return True


# Function to compute the features for a given instance and LP solution
def features(lp, new_sol, types, polyexp=False, orig=None):
    # lp is a gurobi model corresponding to a random instance, new_sol is an np
    # array with the value of the variables in the lp relaxation, & identifier
    # is the integer to enumerate all random instances of a problem

    matrixA = lp.getA()
    matrixA = matrixA.toarray()

    constraints = lp.getConstrs()
    senses = lp.getAttr("Sense")

    if orig is not None:
        matrixA = matrixA[:orig, :]
        constraints = constraints[:orig]
        senses = senses[:orig]

    variables = lp.getVars()
    bl = []
    bu = []

    non_trivial_lb = []
    for i in range(len(variables)):
        if variables[i].LB > 1e-6:
            non_trivial_lb.append(i)
            bl.append(variables[i].LB)
            senses.append(">")
            new_row = np.zeros((1, len(variables)))
            new_row[0, i] = 1
            matrixA = np.vstack([matrixA, new_row])

    non_trivial_ub = []
    for i in range(len(variables)):
        if variables[i].UB < 1e6:
            non_trivial_ub.append(i)
            bu.append(variables[i].UB)
            senses.append("<")
            new_row = np.zeros((1, len(variables)))
            new_row[0, i] = 1
            matrixA = np.vstack([matrixA, new_row])

    ids = np.array(list(range(matrixA.shape[0])))
    non_valid_feature = -1e8 * np.ones(matrixA.shape[0])

    rhs = [constraints[i].RHS for i in range(len(constraints))] + bl + bu

    slacks = np.array(
        [constraint.Slack for constraint in constraints]
        + [
            bl[i] - new_sol[non_trivial_lb[i]]
            for i in range(len(non_trivial_lb))
        ]
        + [
            bu[i] - new_sol[non_trivial_ub[i]]
            for i in range(len(non_trivial_ub))
        ]
    )
    active = np.array(
        [1 if abs(slacks[i]) < 0.001 else 0 for i in range(len(slacks))]
    )
    duals = np.array(
        [constraint.Pi for constraint in constraints]
        + [0] * len(non_trivial_lb)
        + [0] * len(non_trivial_ub)
    )
    sense_eq = np.array(
        [1 if constraint.Sense == "=" else 0 for constraint in constraints]
        + [0] * len(non_trivial_lb)
        + [0] * len(non_trivial_ub)
    )
    sense_leq = np.array(
        [1 if constraint.Sense == "<" else 0 for constraint in constraints]
        + [0] * len(non_trivial_lb)
        + [1] * len(non_trivial_ub)
    )

    constraint_degree = (matrixA != 0).sum(axis=1)

    zero_rhs = np.array(
        [1 if abs(constraint.RHS) < 1e-6 else 0 for constraint in constraints]
        + [0] * len(non_trivial_lb)
        + [0] * len(non_trivial_ub)
    )

    # dataset = np.c_[
    #     problem_id,
    #     ids,
    #     slacks,
    #     active,
    #     duals,
    #     sense_eq,
    #     sense_leq,
    #     constraint_degree,
    #     zero_rhs,
    # ]
    dataset = np.c_[
        slacks, active, duals, sense_eq, sense_leq, constraint_degree, zero_rhs
    ]

    # Global stats
    coef_mean = matrixA.mean(axis=1)
    coef_std = matrixA.std(axis=1)
    coef_min = matrixA.min(axis=1)
    coef_max = matrixA.max(axis=1)
    max_ratio = np.array(
        [
            (
                matrixA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * matrixA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).max(axis=1)
    min_ratio = np.array(
        [
            (
                matrixA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * matrixA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).min(axis=1)
    mean_ratio = np.array(
        [
            (
                matrixA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * matrixA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).mean(axis=1)
    std_ratio = np.array(
        [
            (
                matrixA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * matrixA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).std(axis=1)

    dataset = np.c_[
        dataset,
        coef_mean,
        coef_std,
        coef_min,
        coef_max,
        max_ratio,
        min_ratio,
        mean_ratio,
        std_ratio,
    ]

    # Things that depend on variables
    nonzeroA = matrixA[:, np.where(abs(new_sol) > 0.001)]
    nonzeroA = nonzeroA[:, 0, :]
    zeroA = matrixA[:, np.where(abs(new_sol) < 0.001)]
    zeroA = zeroA[:, 0, :]
    UBA = matrixA[
        :, np.array([abs(var.X - var.UB) < 0.001 for var in lp.getVars()])
    ]

    # Stats for nonzero vars
    coef_mean_nonzero = nonzeroA.mean(axis=1)
    coef_std_nonzero = nonzeroA.std(axis=1)
    coef_min_nonzero = nonzeroA.min(axis=1)
    coef_max_nonzero = nonzeroA.max(axis=1)
    max_ratio_nonzero = np.array(
        [
            (
                nonzeroA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * nonzeroA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).max(axis=1)
    min_ratio_nonzero = np.array(
        [
            (
                nonzeroA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * nonzeroA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).min(axis=1)
    mean_ratio_nonzero = np.array(
        [
            (
                nonzeroA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * nonzeroA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).mean(axis=1)
    std_ratio_nonzero = np.array(
        [
            (
                nonzeroA[i, :] / rhs[i]
                if rhs[i] != 0
                else [-1e8] * nonzeroA.shape[1]
            )
            for i in range(len(rhs))
        ]
    ).std(axis=1)

    dataset = np.c_[
        dataset,
        coef_mean_nonzero,
        coef_std_nonzero,
        coef_min_nonzero,
        coef_max_nonzero,
        max_ratio_nonzero,
        min_ratio_nonzero,
        mean_ratio_nonzero,
        std_ratio_nonzero,
    ]

    # Stats for zero vars
    if zeroA.shape[1] > 0:
        coef_mean_zero = zeroA.mean(axis=1)
        coef_std_zero = zeroA.std(axis=1)
        coef_min_zero = zeroA.min(axis=1)
        coef_max_zero = zeroA.max(axis=1)
        max_ratio_zero = np.array(
            [
                (
                    zeroA[i, :] / rhs[i]
                    if rhs[i] != 0
                    else [-1e8] * zeroA.shape[1]
                )
                for i in range(len(rhs))
            ]
        ).max(axis=1)
        min_ratio_zero = np.array(
            [
                (
                    zeroA[i, :] / rhs[i]
                    if rhs[i] != 0
                    else [-1e8] * zeroA.shape[1]
                )
                for i in range(len(rhs))
            ]
        ).min(axis=1)
        mean_ratio_zero = np.array(
            [
                (
                    zeroA[i, :] / rhs[i]
                    if rhs[i] != 0
                    else [-1e8] * zeroA.shape[1]
                )
                for i in range(len(rhs))
            ]
        ).mean(axis=1)
        std_ratio_zero = np.array(
            [
                (
                    zeroA[i, :] / rhs[i]
                    if rhs[i] != 0
                    else [-1e8] * zeroA.shape[1]
                )
                for i in range(len(rhs))
            ]
        ).std(axis=1)
        dataset = np.c_[
            dataset,
            coef_mean_zero,
            coef_std_zero,
            coef_min_zero,
            coef_max_zero,
            max_ratio_zero,
            min_ratio_zero,
            mean_ratio_zero,
            std_ratio_zero,
        ]
    else:
        non_valid_feature = -1e8 * np.ones((matrixA.shape[0], 8))
        dataset = np.c_[dataset, non_valid_feature]

    # Stats for vars at UB
    if UBA.shape[1] > 0:
        coef_mean_UB = UBA.mean(axis=1)
        coef_std_UB = UBA.std(axis=1)
        coef_min_UB = UBA.min(axis=1)
        coef_max_UB = UBA.max(axis=1)
        max_ratio_UB = np.array(
            [
                UBA[i, :] / rhs[i] if rhs[i] != 0 else [-1e8] * UBA.shape[1]
                for i in range(len(rhs))
            ]
        ).max(axis=1)
        min_ratio_UB = np.array(
            [
                UBA[i, :] / rhs[i] if rhs[i] != 0 else [-1e8] * UBA.shape[1]
                for i in range(len(rhs))
            ]
        ).min(axis=1)
        mean_ratio_UB = np.array(
            [
                UBA[i, :] / rhs[i] if rhs[i] != 0 else [-1e8] * UBA.shape[1]
                for i in range(len(rhs))
            ]
        ).mean(axis=1)
        std_ratio_UB = np.array(
            [
                UBA[i, :] / rhs[i] if rhs[i] != 0 else [-1e8] * UBA.shape[1]
                for i in range(len(rhs))
            ]
        ).std(axis=1)
        dataset = np.c_[
            dataset,
            coef_mean_UB,
            coef_std_UB,
            coef_min_UB,
            coef_max_UB,
            max_ratio_UB,
            min_ratio_UB,
            mean_ratio_UB,
            std_ratio_UB,
        ]
    else:
        non_valid_feature = -1e8 * np.ones((matrixA.shape[0], 8))
        dataset = np.c_[dataset, non_valid_feature]

    # Nonzero vars with nonzero coeffs
    nonzero_coeff_x = (
        matrixA[:, np.where(new_sol > 0.001)][:, 0, :] != 0
    ).sum(axis=1)

    dataset = np.c_[dataset, nonzero_coeff_x]

    # Features for cut evaluation (from Wesselman)
    euclidean_distance = slacks / np.array(
        [np.linalg.norm(matrixA[i, :]) for i in range(len(rhs))]
    )
    rel_violation = np.array(
        [slacks[i] / rhs[i] if rhs[i] != 0 else -1e8 for i in range(len(rhs))]
    )
    adj_dist = slacks / np.array(
        [np.linalg.norm(nonzeroA[i, :]) + 1 for i in range(len(rhs))]
    )
    dataset = np.c_[dataset, euclidean_distance, rel_violation, adj_dist]

    # Features using the cost coefficients
    lp_vars = lp.getVars()
    cost_vector = [var.Obj for var in lp_vars]
    cost_mean = []
    cost_max = []
    cost_min = []
    cost_sd = []
    for i in range(len(rhs)):
        row_cost = np.array(
            [
                cost_vector[i]
                for i in list(np.where(abs(matrixA[i, :]) > 1e-4)[0])
            ]
        )
        if len(row_cost) > 0:
            cost_mean.append(np.mean(row_cost))
            cost_max.append(np.max(row_cost))
            cost_min.append(np.min(row_cost))
            cost_sd.append(np.std(row_cost))
        else:
            cost_mean.append(0)
            cost_max.append(0)
            cost_min.append(0)
            cost_sd.append(0)
    cost_mean = np.array(cost_mean)
    cost_max = np.array(cost_max)
    cost_min = np.array(cost_min)
    cost_sd = np.array(cost_sd)
    dataset = np.c_[dataset, cost_mean, cost_max, cost_min, cost_sd]

    # Features using the ranking of cost coefficients
    normalized_costs = abs(np.array(cost_vector)) / np.max(
        abs(np.array(cost_vector))
    )
    top_costs = np.argsort(normalized_costs)[::-1]
    top_1_costs = top_costs[0 : int(0.01 * lp.NumVars)]
    top_5_costs = top_costs[0 : int(0.05 * lp.NumVars)]
    top_10_costs = top_costs[0 : int(0.10 * lp.NumVars)]
    top_20_costs = top_costs[0 : int(0.20 * lp.NumVars)]

    num_in_1 = []
    num_in_5 = []
    num_in_10 = []
    num_in_20 = []

    for i in range(len(rhs)):
        num_in_1.append(
            np.sum(
                [1 if abs(matrixA[i, j]) > 1e-4 else 0 for j in top_1_costs]
            )
        )
        num_in_5.append(
            np.sum(
                [1 if abs(matrixA[i, j]) > 1e-4 else 0 for j in top_5_costs]
            )
        )
        num_in_10.append(
            np.sum(
                [1 if abs(matrixA[i, j]) > 1e-4 else 0 for j in top_10_costs]
            )
        )
        num_in_20.append(
            np.sum(
                [1 if abs(matrixA[i, j]) > 1e-4 else 0 for j in top_20_costs]
            )
        )

    num_in_1 = np.array(num_in_1)
    num_in_5 = np.array(num_in_5)
    num_in_10 = np.array(num_in_10)
    num_in_20 = np.array(num_in_20)
    dataset = np.c_[dataset, num_in_1, num_in_5, num_in_10, num_in_20]

    frac_in_1 = num_in_1 / constraint_degree
    frac_in_5 = num_in_5 / constraint_degree
    frac_in_10 = num_in_10 / constraint_degree
    frac_in_20 = num_in_20 / constraint_degree
    dataset = np.c_[dataset, frac_in_1, frac_in_5, frac_in_10, frac_in_20]

    singleton = np.array(
        [is_singleton(matrixA[i, :]) for i in range(matrixA.shape[0])]
    )

    aggregation = np.array(
        [
            is_aggregation(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    precedence = np.array(
        [
            is_precedence(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    varbound = np.array(
        [
            is_variable_bound(matrixA[i, :], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    setpart = np.array(
        [
            is_setpartition(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    setpack = np.array(
        [
            is_setpack(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    setcover = np.array(
        [
            is_setcover(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    cardinality = np.array(
        [
            is_cardinality(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    invariantknap = np.array(
        [
            is_invariantknap(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    eqknap = np.array(
        [
            is_equationknap(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    binpack = np.array(
        [
            is_binpack(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    knap = np.array(
        [
            is_knap(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    intknap = np.array(
        [
            is_intknap(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    mixedbin = np.array(
        [
            is_mixedbinary(matrixA[i, :], rhs[i], senses[i], types)
            for i in range(matrixA.shape[0])
        ]
    )

    dataset = np.c_[
        dataset,
        singleton,
        aggregation,
        precedence,
        varbound,
        setpart,
        setpack,
        setcover,
        cardinality,
        invariantknap,
        eqknap,
        binpack,
        knap,
        intknap,
        mixedbin,
    ]

    # column_names = [
    #     "instance_id",
    #     "constraint_id",
    #     "slack",
    #     "is_active",
    #     "dual",
    #     "sense_eq",
    #     "sense_leq",
    #     "degree",
    #     "zero_rhs",
    # ]
    column_names = [
        "slack",
        "is_active",
        "dual",
        "sense_eq",
        "sense_leq",
        "degree",
        "zero_rhs",
    ]
    column_names = column_names + [
        "mean_coefficients",
        "std_coefficients",
        "min_coeffs",
        "max_coeffs",
        "max_ratio",
        "min_ratio",
        "mean_ratio",
        "std_ratio",
    ]
    column_names = column_names + [
        "mean_coefficients_non0",
        "std_coefficients_non0",
        "min_coeffs_non0",
        "max_coeffs_non0",
        "max_ratio_non0",
        "min_ratio_non0",
        "mean_ratio_non0",
        "std_ratio_non0",
    ]
    column_names = column_names + [
        "mean_coefficients_0",
        "std_coefficients_0",
        "min_coeffs_0",
        "max_coeffs_0",
        "max_ratio_0",
        "min_ratio_0",
        "mean_ratio_0",
        "std_ratio_0",
    ]
    column_names = column_names + [
        "mean_coefficients_UB",
        "std_coefficients_UB",
        "min_coeffs_UB",
        "max_coeffs_UB",
        "max_ratio_UB",
        "min_ratio_UB",
        "mean_ratio_UB",
        "std_ratio_UB",
    ]
    column_names = column_names + ["nonzero_coeffs_x"]
    column_names = column_names + [
        "euclidean_distance",
        "relative_violation",
        "adjusted_distance",
    ]
    column_names = column_names + [
        "cost_mean",
        "cost_max",
        "cost_min",
        "cost_std",
    ]
    column_names = column_names + [
        "num_in_1_costs",
        "num_in_5_costs",
        "num_in_10_costs",
        "num_in_20_costs",
    ]
    column_names = column_names + [
        "frac_in_1_costs",
        "frac_in_5_costs",
        "frac_in_10_costs",
        "frac_in_20_costs",
    ]
    column_names = column_names + [
        "is_singleton",
        "is_aggregation",
        "is_precedence",
        "is_varbound",
        "is_setpartion",
        "is_setpacking",
        "is_setcovering",
        "is_cardinality",
        "is_invariantknapsack",
        "is_equationknapsack",
        "is_binpacking",
        "is_knapsack",
        "is_integerknapsack",
        "is_mixedbinary",
    ]

    if polyexp:
        poly = PolynomialFeatures(2, interaction_only=True)
        dataset = poly.fit_transform(dataset[:, 2:])
        column_names = poly.get_feature_names(column_names[2:])

    dataset = np.c_[ids, dataset]
    column_names = ["constraint_id"] + column_names

    return dataset, column_names


def compute_epslabels(labels, eps):
    rows = labels.shape[0]
    columns = labels.shape[1]

    abslabels = np.abs(labels)
    all_labels = []

    for i in range(columns):
        label = 0
        for j in range(rows):
            if abslabels[j, i] > eps:
                label = 2
                break
            elif abslabels[j, i] > 1e-6:
                label = 1
                break
            else:
                label = 0
        all_labels.append(label)

    return all_labels


def is_singleton(row):
    if np.count_nonzero(row) == 1:
        return True
    return False


def is_aggregation(row, rhs, sense, types):
    if np.count_nonzero(row) != 2:
        return False
    if sense != "=":
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    if non_zero_coeffs[0] * non_zero_coeffs[1] < 0:
        return False
    return True


def is_precedence(row, rhs, sense, types):
    if np.count_nonzero(row) != 2:
        return False
    if sense == "=":
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    if non_zero_coeffs[0] != non_zero_coeffs[1]:
        return False
    return True


def is_variable_bound(row, sense, types):
    if np.count_nonzero(row) != 2:
        return False
    if sense != "<":
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_types = [types[i] for i in non_zero_indices]
    if non_zero_types.count("B") != 1:
        return False
    return True


def is_setpartition(row, rhs, sense, types):
    if sense != "=":
        return False
    if rhs not in [1, -1]:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    if not np.all(np.isin(non_zero_coeffs, [-1, 1])):
        return False
    return True


def is_setpack(row, rhs, sense, types):
    if sense != "<":
        return False
    if rhs != 1:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    if sum(non_zero_coeffs == 1) != len(non_zero_coeffs):
        return False
    return True


def is_setcover(row, rhs, sense, types):
    if sense != ">":
        return False
    if rhs != 1:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    if sum(non_zero_coeffs == 1) != len(non_zero_coeffs):
        return False
    return True


def is_cardinality(row, rhs, sense, types):
    if sense != "=":
        return False
    if rhs < 2:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    if sum(non_zero_coeffs == 1) != len(non_zero_coeffs):
        return False
    return True


def is_invariantknap(row, rhs, sense, types):
    if sense != "<":
        return False
    if rhs < 2:
        return False
    if abs(int(rhs) - rhs) > 1e-6:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    if sum(non_zero_coeffs == 1) != len(non_zero_coeffs):
        return False
    return True


def is_equationknap(row, rhs, sense, types):
    if sense != "=":
        return False
    if rhs < 2:
        return False
    if abs(int(rhs) - rhs) > 1e-6:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    return True


def is_binpack(row, rhs, sense, types):
    if sense == "=":
        return False
    if rhs < 2:
        return False
    if abs(int(rhs) - rhs) > 1e-6:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_types = [types[i] for i in non_zero_indices]
    non_zero_coeffs = np.array([row[i] for i in non_zero_indices])
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    if rhs not in non_zero_coeffs:
        return False
    return True


def is_knap(row, rhs, sense, types):
    if sense == "=":
        return False
    if rhs < 2:
        return False
    if abs(int(rhs) - rhs) > 1e-6:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types or "I" in non_zero_types:
        return False
    return True


def is_intknap(row, rhs, sense, types):
    if sense == "=":
        return False
    if abs(int(rhs) - rhs) > 1e-6:
        return False
    non_zero_indices = np.nonzero(row)[0]
    non_zero_types = [types[i] for i in non_zero_indices]
    if "C" in non_zero_types:
        return False
    return True


def is_mixedbinary(row, rhs, sense, types):
    non_zero_indices = np.nonzero(row)[0]
    non_zero_types = [types[i] for i in non_zero_indices]
    if "B" not in non_zero_types:
        return False
    if "C" not in non_zero_types:
        return False
    if "I" in non_zero_types:
        return False
    return True
