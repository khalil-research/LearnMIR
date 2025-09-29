##############################################################################
# Learning to Cut. Oscar Guaje, Arnaud Deza, Aleksandr Kazachkov, Elias Khalil
# My implementation of the Appx-MIR-SEP from
# Dash, S., Gunluk, O., Lodi, A.: MIR closures of polyhedral sets. Math.
##############################################################################

import numpy as np
import gurobipy as gp


class Mirsep:
    def __init__(self, ip, solution, k, timelimit):

        self.ip = ip.copy()
        self.b = self.ip.getAttr("rhs")
        self.vars = self.ip.getVars()
        self.cont_idx = [
            idx
            for idx, var in enumerate(self.vars)
            if var.vtype == gp.GRB.CONTINUOUS
        ]
        self.int_idx = [
            idx
            for idx, var in enumerate(self.vars)
            if idx not in self.cont_idx
        ]
        self.matrix = self.ip.getA().toarray()
        self.standardize()
        self.C = self.matrix[:, self.cont_idx]
        self.A = self.matrix[:, self.int_idx]
        self.D = self.matrix[:, self.ip.NumVars:]

        self.x_star = None
        self.v_star = None
        self.s_star = None
        self.slacks = None
        self.update_solution(solution)

        self.model = None
        self.timelimit = timelimit
        self.log = None
        # self.epsilon = 1e-4
        self.epsilon = 0
        self.k = k
        self.k_set = [1 / (2**idx) for idx in range(1, self.k + 1)]
        self.lambd = None
        self.c_plus = None
        self.alpha_hat = None
        self.alpha_bar = None
        self.beta_hat = None
        self.beta_bar = None
        self.pi = None
        self.delta = None
        self.delta_k = None

    def build_model(self, logfile, predictions=None):
        self.model = gp.Model("mirsep")

        self.model.ModelSense = gp.GRB.MAXIMIZE
        self.model.Params.PoolSolutions = 1000
        self.model.Params.TimeLimit = self.timelimit
        self.model.Params.LogToConsole = 1
        self.model.Params.LogFile = logfile

        self.lambd = self.model.addVars(
            self.matrix.shape[0],
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
            obj=0,
            name="lambda",
        )

        if predictions is not None:
            fixed_count = 0
            for idx, prediction in enumerate(predictions):
                if prediction <= 1e-6:
                    fixed_count += 1
                    self.lambd[idx].LB = 0
                    self.lambd[idx].UB = 0
            print("Fixed ", fixed_count, "lambdas to 0")

        self.non_zero_v = [
            idx for idx in range(len(self.v_star)) if self.v_star[idx] > 0
        ]
        self.c_plus = self.model.addVars(
            self.non_zero_v,
            obj=[-self.v_star[idx] for idx in self.non_zero_v],
            name="c_plus",
        )

        self.non_zero_s = [
            idx for idx in range(len(self.s_star)) if self.s_star[idx] > 0
        ]
        self.c_plus_slack = self.model.addVars(
            self.non_zero_s,
            obj=[-self.s_star[idx] for idx in self.non_zero_s],
            name="c_plus_slack",
        )

        self.non_zero_x = [
            idx for idx in range(len(self.x_star)) if self.x_star[idx] > 0
        ]
        self.alpha_bar = self.model.addVars(
            self.non_zero_x,
            lb=-gp.GRB.INFINITY,
            vtype=gp.GRB.INTEGER,
            name="alpha_int",
        )
        self.alpha_hat = self.model.addVars(
            self.non_zero_x,
            ub=1 - self.epsilon,
            obj=[-self.x_star[idx] for idx in self.non_zero_x],
            name="alpha_cont",
        )

        self.beta_bar = self.model.addVar(
            lb=-gp.GRB.INFINITY, vtype=gp.GRB.INTEGER, name="beta_int"
        )
        self.beta_hat = self.model.addVar(
            ub=1 - self.epsilon, name="beta_cont"
        )

        self.pi = self.model.addVars(
            self.k_set, vtype=gp.GRB.BINARY, name="pi_k"
        )
        self.delta = self.model.addVar(name="delta")
        self.delta_k = self.model.addVars(
            self.k_set, obj=self.k_set, name="delta_k"
        )

        self.model.addConstrs(
            (
                self.c_plus[idx]
                >= gp.quicksum(
                    [
                        self.C[i, idx] * self.lambd[i]
                        for i in range(self.matrix.shape[0])
                    ]
                )
                for idx in self.non_zero_v
            ),
            name="cont_PI",
        )
        self.model.addConstrs(
            (
                self.c_plus_slack[idx]
                >= gp.quicksum(
                    [
                        self.D[i, idx] * self.lambd[i]
                        for i in range(self.matrix.shape[0])
                    ]
                )
                for idx in self.non_zero_s
            ),
            name="slack_PI",
        )
        self.model.addConstrs(
            (
                self.alpha_hat[idx] + self.alpha_bar[idx]
                >= gp.quicksum(
                    [
                        self.A[i, idx] * self.lambd[i]
                        for i in range(self.matrix.shape[0])
                    ]
                )
                for idx in self.non_zero_x
            ),
            name="alpha_PI",
        )
        self.model.addConstr(
            self.beta_hat + self.beta_bar
            <= gp.quicksum(
                [
                    self.b[i] * self.lambd[i]
                    for i in range(self.matrix.shape[0])
                ]
            ),
            name="beta_PI",
        )

        self.model.addConstr(
            self.beta_hat
            >= gp.quicksum([eps * self.pi[eps] for eps in self.k_set]),
            name="beta_hat_repr",
        )
        self.model.addConstr(
            self.delta
            == (self.beta_bar + 1)
            - gp.quicksum(
                [
                    self.alpha_bar[idx] * self.x_star[idx]
                    for idx in self.non_zero_x
                ]
            ),
            name="delta_val",
        )
        self.model.addConstrs(
            (self.delta_k[idx] <= self.delta for idx in self.k_set),
            name="delta_k_rel",
        )
        self.model.addConstrs(
            (self.delta_k[idx] <= self.pi[idx] for idx in self.k_set),
            name="pi_k_rel",
        )

        self.model.update()

    def standardize(self):
        for idx, constraint in enumerate(self.ip.getConstrs()):
            if constraint.sense == "<":
                ones = [1 if i == idx else 0 for i in range(len(self.b))]
                self.matrix = np.c_[self.matrix, ones]
            elif constraint.sense == ">":
                ones = [-1 if i == idx else 0 for i in range(len(self.b))]
                self.matrix = np.c_[self.matrix, ones]
        for idx, var in enumerate(self.ip.getVars()):
            if var.LB > 1e-6:
                ones = [
                    1 if i == idx else 0 for i in range(self.matrix.shape[1])
                ]
                self.matrix = np.r_[self.matrix, [np.array(ones)]]
                self.b.append(var.LB)
                ones = np.zeros(len(self.b))
                ones[-1] = -1
                self.matrix = np.c_[self.matrix, ones]
        for idx, var in enumerate(self.ip.getVars()):
            if var.UB < 1e6:
                ones = [
                    1 if i == idx else 0 for i in range(self.matrix.shape[1])
                ]
                self.matrix = np.r_[self.matrix, [np.array(ones)]]
                self.b.append(var.UB)
                ones = np.zeros(len(self.b))
                ones[-1] = 1
                self.matrix = np.c_[self.matrix, ones]

    def get_gmi_cut(self, solution):
        cut = []
        return cut

    def update_solution(self, solution):
        self.v_star = [solution[idx] for idx in self.cont_idx]
        # print("v")
        # for s in range(len(self.v_star)):
        #     if self.v_star[s] < 0:
        #         print(self.v_star[s])
        self.x_star = [solution[idx] for idx in self.int_idx]
        # print("x")
        # for s in range(len(self.x_star)):
        #     if self.x_star[s] < 0:
        #         print(self.x_star[s])
        temp_D_rows = [i for i in range(len(self.b)) if np.any(self.D[i, :])]
        temp_D = self.D[temp_D_rows, :]
        self.slacks = (
            self.b - np.dot(self.A, self.x_star) - np.dot(self.C, self.v_star)
        )
        self.s_star = np.dot(
            np.linalg.inv(temp_D),
            np.array(
                self.b
                - np.dot(self.A, self.x_star)
                - np.dot(self.C, self.v_star)
            )[temp_D_rows],
        )
        # print("s")
        # for s in range(len(self.s_star)):
        #     if self.s_star[s] < 0:
        #         print(self.s_star[s])

    def solve(self, write=False):
        if write:
            self.model.write("model.lp")
        self.model.optimize()

    def get_cuts(self):
        # print(self.model.ObjVal)
        self.cuts = []
        if self.model.Status in [2, 9]:
            num_sols = self.model.SolCount
            for sol_idx in range(num_sols):
                self.cuts.append(self.compute_cut(sol_idx))

        return self.cuts

    def get_lambdas(self):
        self.lambdas = []
        if self.model.Status in [2, 9]:
            num_sols = self.model.SolCount
            for sol_idx in range(num_sols):
                self.lambdas.append(self.compute_lambda(sol_idx))

        return self.lambdas

    def get_best_cut(self):
        c_pos = np.dot([self.lambd[i].X for i in range(len(self.b))], self.C)
        c_pos = np.maximum(0, c_pos)

        alpha = np.dot([self.lambd[i].X for i in range(len(self.b))], self.A)
        for i in range(len(alpha)):
            if abs(alpha[i]) < 1e-6:
                alpha[i] = 0
        # print(alpha)
        alpha_int = np.floor(alpha)
        alpha_frac = alpha - alpha_int

        coeffs = np.zeros(self.ip.NumVars)
        for idx, original_idx in enumerate(self.int_idx):
            coeffs[original_idx] = (
                alpha_frac[idx] + self.beta_hat.X * alpha_int[idx]
            )
        for idx, original_idx in enumerate(self.cont_idx):
            coeffs[original_idx] = c_pos[idx]
        coeffs = np.append(coeffs, self.beta_hat.X * (self.beta_bar.X + 1))
        # print(coeffs)
        return coeffs

    def compute_lambda(self, index):
        self.model.Params.SolutionNumber = index
        this_lambda = [self.lambd[i].Xn for i in range(self.matrix.shape[0])]
        return this_lambda

    def compute_cut(self, index, debug_print=False):
        self.model.Params.SolutionNumber = index
        c_pos = np.dot([self.lambd[i].Xn for i in range(len(self.b))], self.C)
        c_pos = np.maximum(0, c_pos)

        # if debug_print:
        #     for i in self.non_zero_v:
        #         if c_pos[i] != self.c_plus[i].Xn:
        #             print(c_pos[i], self.c_plus[i].Xn)

        alpha = np.dot([self.lambd[i].Xn for i in range(len(self.b))], self.A)
        for i in range(len(alpha)):
            if abs(alpha[i]) < 1e-6:
                # if debug_print:
                #     print(i, i in self.non_zero_x, alpha[i])
                alpha[i] = 0
        # print(alpha)
        alpha_int = np.floor(alpha)
        alpha_frac = alpha - alpha_int
        # if debug_print:
        # print(self.non_zero_x)
        # for i in self.non_zero_x:
        # print(
        #     i,
        #     alpha_int[i],
        #     alpha_frac[i],
        #     alpha[i],
        #     self.alpha_bar[i].Xn,
        #     self.alpha_hat[i].Xn,
        # )

        coeffs = np.zeros(self.ip.NumVars)
        for idx, original_idx in enumerate(self.int_idx):
            coeffs[original_idx] = (
                alpha_frac[idx] + self.beta_hat.Xn * alpha_int[idx]
            )
        for idx, original_idx in enumerate(self.cont_idx):
            coeffs[original_idx] = c_pos[idx]
        coeffs = np.append(coeffs, self.beta_hat.Xn * (self.beta_bar.Xn + 1))
        # beta = np.dot([self.lambd[i].Xn for i in range(len(self.b))], self.b)
        # if debug_print:
        # print(self.beta_hat.Xn, self.beta_bar.Xn, beta, self.delta.Xn)
        # print(coeffs)
        return coeffs
