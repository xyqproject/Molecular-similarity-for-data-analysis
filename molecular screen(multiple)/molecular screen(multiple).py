import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gams import GamsWorkspace
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
from sklearn.svm import SVR
import pickle
import warnings
from skopt.space import Space
from skopt.sampler import Grid
warnings.filterwarnings("ignore")
from pyomo.environ import *



def find_center(lst_valency, lst_Weight, selected_groups, selected_properties, cluster_molecule):
    # build model
    model = pyo.ConcreteModel()
    model.i1 = pyo.Set(initialize=selected_groups)
    model.j1 = pyo.Set(initialize=list(cluster_molecule.index))
    # read parameters
    model.Noc = pyo.Param(model.i1, initialize=dict(
        zip(selected_groups, [lst_valency.loc[item, "Noc"] for item in selected_groups])))
    model.Mw = pyo.Param(model.i1, initialize=dict(
        zip(selected_groups, [lst_Weight.loc[item, "Mw"] for item in selected_groups])))
    model.nt = pyo.Param(model.j1, model.i1,
                         initialize={(j, i): cluster_molecule.loc[j][i] for i in model.i1 for j in model.j1})


    model.R0 = pyo.Param(initialize=8.314)  # ideal gas index [J/mol/K]
    model.T0 = pyo.Param(initialize=298.15)  # Temperature [K]
    model.EP = pyo.Param(initialize=1e-6)  # threshold
    model.M = pyo.Param(initialize=1000)  # big M

    model.TotGrp_U = pyo.Param(initialize=selected_properties["All groups"][1])  # group total UB
    model.TotGrp_L = pyo.Param(initialize=selected_properties["All groups"][0])  # group total LB
    model.FuncGrp_U = pyo.Param(initialize=selected_properties["All functional groups"][1])  # functional group total UB
    model.FuncGrp_L = pyo.Param(initialize=selected_properties["All functional groups"][0])  # functional group total LB
    model.SameGrp_U = pyo.Param(initialize=selected_properties["The same groups"][1])  # same group UB
    model.Mw_U = pyo.Param(initialize=selected_properties["Molecular Weight"][1])  # molecular weight UB [g/Mol]
    model.Mw_L = pyo.Param(initialize=selected_properties["Molecular Weight"][0])  # molecular weight LB [g/Mol]
    # variables
    model.n1 = pyo.Var(model.i1, domain=pyo.NonNegativeIntegers, bounds=(0, model.SameGrp_U))
    model.q = pyo.Var(domain=pyo.Integers,
                      bounds=(selected_properties["q"][0], selected_properties["q"][1]))  # no ring
    model.y = pyo.Var(model.j1, model.i1, domain=pyo.Binary)
    model.union = pyo.Var(model.j1, model.i1, domain=pyo.NonNegativeIntegers, bounds=(0, model.SameGrp_U))
    model.intersection = pyo.Var(model.j1, model.i1, domain=pyo.NonNegativeIntegers, bounds=(0, model.SameGrp_U))
    model.aveMSC = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.MSC = pyo.Var(model.j1, domain=pyo.NonNegativeReals, bounds=(0, 1))

    # define constraints

    def n_g_l_rule(model):
        return sum(model.n1[i] for i in model.i1) >= model.TotGrp_L

    model.n_g_l = pyo.Constraint(rule=n_g_l_rule)

    def n_g_u_rule(model):
        return sum(model.n1[i] for i in model.i1) <= model.TotGrp_U

    model.n_g_u = pyo.Constraint(rule=n_g_u_rule)

    def n_f_l_rule(model):
        non_func_groups = ['CH3', 'CH2', 'CH', 'C']
        return (sum(model.n1[i] for i in model.i1) -
                sum(model.n1[g] for g in non_func_groups if g in model.i1)) >= model.FuncGrp_L

    model.n_f_l = pyo.Constraint(rule=n_f_l_rule)

    def n_f_u_rule(model):
        non_func_groups = ['CH3', 'CH2', 'CH', 'C']
        return (sum(model.n1[i] for i in model.i1) -
                sum(model.n1[g] for g in non_func_groups if g in model.i1)) <= model.FuncGrp_U

    model.n_f_u = pyo.Constraint(rule=n_f_u_rule)

    def val_0_rule(model):
        return sum((2 - model.Noc[i]) * model.n1[i] for i in model.i1) == 2 * model.q

    model.val_0 = pyo.Constraint(rule=val_0_rule)

    def val_1_rule(model, i):
        return sum(model.n1[j] for j in model.i1 if j != i) >= (model.Noc[i] - 2) * model.n1[i] + 2

    model.val_1 = pyo.Constraint(model.i1, rule=val_1_rule)

    def mw_up_rule(model):
        return sum(model.n1[i] * model.Mw[i] for i in model.i1) <= model.Mw_U

    model.mw_up = pyo.Constraint(rule=mw_up_rule)

    def mw_low_rule(model):
        return sum(model.n1[i] * model.Mw[i] for i in model.i1) >= model.Mw_L

    model.mw_low = pyo.Constraint(rule=mw_low_rule)

    def cons_aug_1(model, j, i):
        return model.n1[i] - model.nt[j, i] <= (1 - model.y[j, i]) * model.M

    model.cons_aug_1 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_1)

    def cons_aug_2(model, j, i):
        return model.nt[j, i] - model.n1[i] <= model.y[j, i] * model.M - 1

    model.cons_aug_2 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_2)

    def cons_aug_3(model, j, i):
        return model.intersection[j, i] <= model.n1[i]

    model.cons_aug_3 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_3)

    def cons_aug_4(model, j, i):
        return model.intersection[j, i] <= model.nt[j, i]

    model.cons_aug_4 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_4)

    def cons_aug_7(model, j, i):
        return model.intersection[j, i] >= model.n1[i] - (1 - model.y[j, i]) * model.M

    model.cons_aug_7 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_7)

    def cons_aug_8(model, j, i):
        return model.intersection[j, i] >= model.nt[j, i] - model.y[j, i] * model.M

    model.cons_aug_8 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_8)

    def cons_aug_5(model, j, i):
        return model.union[j, i] >= model.n1[i]

    model.cons_aug_5 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_5)

    def cons_aug_6(model, j, i):
        return model.union[j, i] >= model.nt[j, i]

    model.cons_aug_6 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_6)

    def cons_aug_9(model, j, i):
        return model.union[j, i] <= model.n1[i] + model.y[j, i] * model.M

    model.cons_aug_9 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_9)

    def cons_aug_10(model, j, i):
        return model.union[j, i] <= model.nt[j, i] + (1 - model.y[j, i]) * model.M

    model.cons_aug_10 = pyo.Constraint(model.j1, model.i1, rule=cons_aug_10)

    def cons_aug_11(model, j):
        return model.MSC[j] * (prod((model.union[j, i] + 1) for i in model.i1) - 1) == (
                    prod((model.intersection[j, i] + 1) for i in model.i1) - 1)

    model.cons_aug_11 = pyo.Constraint(model.j1, rule=cons_aug_11)


    def cons_aug_12(model, j):
        return model.MSC[j] <=0.99

    model.cons_aug_12 = pyo.Constraint(model.j1, rule=cons_aug_12)



    def cons_aug_13(model):
        return model.aveMSC * len(cluster_molecule.index) == sum(model.MSC[j] for j in model.j1)

    model.cons_aug_13 = pyo.Constraint(rule=cons_aug_13)

    model.obj = pyo.Objective(expr=model.aveMSC, sense=pyo.maximize)

    solver = SolverFactory('gams')
    solver.options["solver"] = "baron"
    results = solver.solve(model, tee=True, keepfiles=False)


    return list(value(model.MSC[j]) for j in model.j1),list(value(model.n1[i]) for i in model.i1)



path="\molecular screen(multiple)"
df=pd.read_excel(fr"{path}\dataset\data test for n_to_1.xlsx",index_col=0)
file_path_group_info = fr"{path}\dataset\group_Noc_Mw.xlsx"
df_valency = pd.read_excel(file_path_group_info, sheet_name="valency", index_col=0)
df_weight = pd.read_excel(file_path_group_info, sheet_name="weight", index_col=0)

lst_group = list(df.columns)
lst_structural_constraints = {'Molecular Weight': (0.0, 200.0), 'The same groups': (0.0, 3),
                              'All groups': (2.0, 8.0), 'All functional groups': (0.0, 3.0), 'q': (1.0, 1.0)}




for _ in range(5):
    MSCs,res=find_center(df_valency, df_weight, lst_group, lst_structural_constraints, df)
    print(MSCs)
    print(res)
    newline=pd.DataFrame([res],columns=df.columns)
    df=pd.concat([df,newline],ignore_index=True)
print(df)
df.to_excel(fr"{path}\result\result for multiple reference.xlsx")


