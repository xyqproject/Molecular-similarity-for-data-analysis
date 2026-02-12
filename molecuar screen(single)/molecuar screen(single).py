from gams.core import gdx
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



def create_milp_model(lst_group,lst_valency,lst_Weight,selected_groups,selected_properties,reference_molecule,MSC_low):
    # build model
    model = pyo.ConcreteModel()
    model.i1 = pyo.Set(initialize=selected_groups)
    # read parameter
    model.Noc = pyo.Param(model.i1,initialize=dict(zip(selected_groups,[lst_valency.loc[item,"Noc"] for item in selected_groups])))
    model.Mw = pyo.Param(model.i1,initialize=dict(zip(selected_groups,[lst_Weight.loc[item,"Mw"] for item in selected_groups])))
    model.nt=pyo.Param(model.i1,initialize=dict(zip(model.i1,reference_molecule)))

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
    model.q = pyo.Var(domain=pyo.Integers, bounds=(selected_properties["q"][0], selected_properties["q"][1]))  # no ring
    model.dummy = pyo.Var()
    model.y=pyo.Var(model.i1,domain=pyo.Binary)
    model.union=pyo.Var(model.i1,domain=pyo.NonNegativeIntegers,bounds=(0, model.SameGrp_U))
    model.intersection=pyo.Var(model.i1,domain=pyo.NonNegativeIntegers,bounds=(0, model.SameGrp_U))

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
        # 减去非功能基团
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
    def edummy_rule(model):
        return model.dummy == 1
    model.edummy = pyo.Constraint(rule=edummy_rule)
    model.obj = pyo.Objective(expr=model.dummy, sense=pyo.minimize)



    def cons_aug_1(model,i):
        return model.n1[i]-model.nt[i]<=(1-model.y[i])*model.M
    model.cons_aug_1=pyo.Constraint(model.i1,rule=cons_aug_1)
    def cons_aug_2(model,i):
        return model.nt[i]-model.n1[i]<=model.y[i]*model.M-1
    model.cons_aug_2=pyo.Constraint(model.i1,rule=cons_aug_2)

    def cons_aug_3(model,i):
        return model.intersection[i]<=model.n1[i]
    model.cons_aug_3=pyo.Constraint(model.i1,rule=cons_aug_3)
    def cons_aug_4(model,i):
        return model.intersection[i]<=model.nt[i]
    model.cons_aug_4=pyo.Constraint(model.i1,rule=cons_aug_4)
    def cons_aug_7(model,i):
        return model.intersection[i]>=model.n1[i]-(1-model.y[i])*model.M
    model.cons_aug_7=pyo.Constraint(model.i1,rule=cons_aug_7)
    def cons_aug_8(model,i):
        return model.intersection[i]>=model.nt[i]-model.y[i]*model.M
    model.cons_aug_8=pyo.Constraint(model.i1,rule=cons_aug_8)


    def cons_aug_5(model,i):
        return model.union[i]>=model.n1[i]
    model.cons_aug_5=pyo.Constraint(model.i1,rule=cons_aug_5)
    def cons_aug_6(model,i):
        return model.union[i]>=model.nt[i]
    model.cons_aug_6=pyo.Constraint(model.i1,rule=cons_aug_6)
    def cons_aug_9(model,i):
        return model.union[i]<=model.n1[i]+model.y[i]*model.M
    model.cons_aug_9=pyo.Constraint(model.i1,rule=cons_aug_9)
    def cons_aug_10(model,i):
        return model.union[i]<=model.nt[i]+(1-model.y[i])*model.M
    model.cons_aug_10=pyo.Constraint(model.i1,rule=cons_aug_10)

    def cons_aug_11(model):
        return (prod((model.intersection[i]+1) for i in model.i1)-1)>=MSC_low*(prod((model.union[i]+1) for i in model.i1)-1)
    model.cons_aug_11=pyo.Constraint(rule=cons_aug_11)


    return model


def solve_with_configured_baron(model):
    solver = SolverFactory('gams')
    baron_options = [
        f"numsol 1000",
        "gdxout baron_solutions"
    ]

    gams_commands = ["$onecho > baron.opt"  ]
    gams_commands.extend(baron_options)
    gams_commands.append("$offecho")
    gams_commands.extend([
        "option reslim=1e6;"
    ])
    gams_commands.extend(["GAMS_MODEL.optfile = 1;"])
    solver.options["solver"] = "baron"
    pathtemp = './results'
    solver.options["tmpdir"] = pathtemp
    solver.options["add_options"] = gams_commands
    results = solver.solve(model,tee=False,keepfiles=True)

    return results.solution

def molecular_generation(lst_group,lst_valency,lst_Weight,selected_groups,selected_properties,reference_molecule,MSC_low):
    model = create_milp_model(lst_group,lst_valency,lst_Weight,selected_groups,selected_properties,reference_molecule,MSC_low)
    res = solve_with_configured_baron(model)
    gdx_h = gdx.new_gdxHandle_tp()
    ws=GamsWorkspace()
    df = pd.DataFrame(columns=selected_groups)
    for i in range(1000):
        file_path=fr"{os.getcwd()}\results\baron_solutions{i+1}.gdx"
        if not os.path.exists(file_path):
            break
        db=ws.add_database_from_gdx(file_path)
        for j in range(len(selected_groups)):
            df.loc[i,selected_groups[j]]=db[f"x{j+1}"].find_record().level
        db=None
        os.remove(file_path)
    return df





path="\molecuar screen(single)"


file_path_group_info=fr"{path}\dataset\group_Noc_Mw.xlsx"
df_group = pd.read_excel(file_path_group_info,sheet_name="group", index_col=0)
df_valency = pd.read_excel(file_path_group_info, sheet_name="valency",index_col=0)
df_weight = pd.read_excel(file_path_group_info, sheet_name="weight",index_col=0)


list_group=['CH3','CH2','CH','C','COOH','OH','COO']
print(list_group)


reference_molecule=[2,14,1,0,1,0,0]
print(reference_molecule)
list_structural_constraints={'Molecular Weight': (0.0, 400.0), 'The same groups': (0.0, 20), 'All groups': (2.0, 30.0), 'All functional groups': (0.0, 5.0), 'q': (1.0, 1.0)}
MSC_low=0.7
df_step1_out=molecular_generation(df_group,df_valency,df_weight,list_group,list_structural_constraints,reference_molecule,MSC_low)
print(df_step1_out)

df_step1_out.to_excel(fr"{path}\results\generated molecule.xlsx")



#calculate MSC to reference molecules............................................
def JSC(list1, list2):  # index3+index2
    intersection = [min(list1[i], list2[i]) + 1 for i in range(len(list1))]
    union = [max(list1[i], list2[i]) + 1 for i in range(len(list1))]
    res = (np.prod(intersection) - 1) / (np.prod(union) - 1)
    return res

for i in df_step1_out.index:
    m1=df_step1_out.loc[i,list_group].values
    msc=JSC(m1,reference_molecule)
    df_step1_out.loc[i,"MSC to target"]=msc
df_step1_out.to_excel(fr"{path}\results\generated molecule(with MSC to target).xlsx")





