import pandas as pd
import numpy as np
from pyomo.opt import SolverFactory
from pyomo.environ import *
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle
import warnings
from skopt.space import Space
from skopt.sampler import Grid
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
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


path=r"\data augmentation\dataset"

# .................................................molecular augmentation...............................................
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




# .................................................training info generation.............................................

df_group=pd.read_excel(fr"{path}\220group.xlsx",index_col=0)
property="vc"
df = pd.read_excel(fr"{path}\data test for augmentation (after correct and augmentation).xlsx", index_col=0)
train_num = int(1*len(df.index))
molecule=[df.loc[j, [f"Group {i + 1}" for i in range(220)]].values for j in range(1,train_num+1)]
n = 10



def JSC(list1,list2):     #index3+index2
    intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
    union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
    res=(np.product(intersection)-1)/(np.product(union)-1)
    return res

def Linear1(molecule_new,molecule):
    JSC_dict={i+1:JSC(molecule_new,molecule[i]) for i in range(len(molecule))}
    sorted_JSC = {key: JSC_dict[key] for key in sorted(JSC_dict.keys(), key=JSC_dict.get,  reverse=True)[1:]}
    molecule_training=[df.loc[list(sorted_JSC.keys())[i], [f"Group {j + 1}" for j in range(220)]].values for i in range(n)]



    property_training=[df.loc[list(sorted_JSC.keys())[i], f"{property}"] for i in range(n)]
    print({key: JSC_dict[key] for key in list(sorted_JSC.keys())[0:n]})
    print(property_training)

    df_training=pd.DataFrame()
    df_training.loc[0, "JSC"] = 1
    df_training.loc[0, f"{property}"] = 9999
    df_training.loc[0, "original_No"] = 9999
    for j in range(220):
        if molecule_new[j] > 0:
            df_training.loc[0, df_group.iloc[j, 0]] = molecule_new[j]

    for i in range(n):
        df_training.loc[i+1,"JSC"]=JSC_dict[list(sorted_JSC.keys())[i]]
        df_training.loc[i+1,f"{property}"]=df.loc[list(sorted_JSC.keys())[i], f"{property}"]
        df_training.loc[i+1,"original_No"]=list(sorted_JSC.keys())[i]
        for j in range(220):
            if df.loc[list(sorted_JSC.keys())[i], f"Group {j + 1}"]>0:
                df_training.loc[i+1,df_group.iloc[j,0]]=df.loc[list(sorted_JSC.keys())[i], f"Group {j + 1}"]


    model = ConcreteModel()
    model.I = Set(initialize=[i for i in range(220)])
    model.par = Var(model.I, within=Reals, initialize=0)
    model.par0 = Var(within=Reals, initialize=0)

    def obj_rule(model):
        error = 0
        for i in range(len(molecule_training)):
            Tc = model.par0
            for j in range(220):
                Tc += model.par[j] * molecule_training[i][j]
            error += (Tc - property_training[i]) ** 2
        error = (error / len(molecule_training)) ** 0.5
        return error

    model.obj = Objective(rule=obj_rule, sense=minimize)

    opt = SolverFactory('gams')
    io_options = dict()
    io_options['solver'] = "minos"
    io_options['mtype'] = "NLP"
    result = opt.solve(model, tee=False, keepfiles=False, io_options=io_options)

    pro_pre = np.sum([value(model.par[i]) * molecule_new[i] for i in range(220)]) + value(model.par0)

    print(value(model.par0))

    for group in df_training.columns[3:]:
        for j in range(220):
            if df_group.iloc[j, 0]==group:
                df_training.loc[n + 1, group] = value(model.par[j])


    df_training = df_training.iloc[:-1, 3:].reset_index(drop=True)
    df_training.index += 1
    return pro_pre,list(sorted_JSC.values())[0],df_training



list_df=[]
num_augmentation=10
for i in df.index:

    number=i
    print(i)
    molecule_new = df.loc[i, [f"Group {j + 1}" for j in range(220)]].values
    property_new=df.loc[i, f"{property}"]

    property_pre,JSC_new,df_info=Linear1(molecule_new, molecule)
    if JSC_new>=0.9:

        print(df_info)


        file_path_group_info = fr"{path}\group_Noc_Mw.xlsx"
        df_valency = pd.read_excel(file_path_group_info, sheet_name="valency", index_col=0)
        df_weight = pd.read_excel(file_path_group_info, sheet_name="weight", index_col=0)

        lst_group = list(df_info.columns)
        lst_structural_constraints = {'Molecular Weight': (0.0, 1000.0), 'The same groups': (0.0, 50),
                                      'All groups': (2.0, 50.0), 'All functional groups': (0.0, 20.0), 'q': (1.0, 1.0)}


        for _ in range(num_augmentation):
            MSCs, res = find_center(df_valency, df_weight, lst_group, lst_structural_constraints, df_info)
            print(MSCs)
            print(res)
            newline = pd.DataFrame([res], columns=df_info.columns)
            df_info = pd.concat([df_info, newline], ignore_index=True)
        print(df_info)


        property = "vc"
        df_step1_out = df_info
        df_group = pd.read_excel(fr"{path}\220group.xlsx", index_col=0)


        df_step2_out = pd.DataFrame(index=[i + 1 for i in range(num_augmentation)], columns=df_group["Group"].values)

        for i in range(num_augmentation):
            ii = list(df_step1_out.index)[-num_augmentation] + i
            for j in df_step1_out.columns:
                if df_step1_out.loc[ii, j] != "NaN":
                    df_step2_out.loc[i + 1, j] = df_step1_out.loc[ii, j]
        df_step2_out = df_step2_out.fillna(0)
        df_step2_out.columns = [f"Group {i + 1}" for i in range(220)]







        df = pd.read_excel(fr"{path}\data test for augmentation (after correct and augmentation).xlsx", index_col=0)
        train_num = int(1 * len(df.index))
        molecule = [df.loc[j, [f"Group {i + 1}" for i in range(220)]].values for j in range(1, train_num + 1)]
        n = 10


        def JSC(list1, list2):  # index3+index2
            intersection = [min(list1[i], list2[i]) + 1 for i in range(len(list1))]
            union = [max(list1[i], list2[i]) + 1 for i in range(len(list1))]
            res = (np.product(intersection) - 1) / (np.product(union) - 1)
            return res


        def Linear2(molecule_new, molecule):
            JSC_dict = {i + 1: JSC(molecule_new, molecule[i]) for i in range(len(molecule))}
            sorted_JSC = {key: JSC_dict[key] for key in sorted(JSC_dict.keys(), key=JSC_dict.get, reverse=True)[0:]}
            molecule_training = [df.loc[list(sorted_JSC.keys())[i], [f"Group {j + 1}" for j in range(220)]].values for i
                                 in range(n)]



            property_training = [df.loc[list(sorted_JSC.keys())[i], f"{property}"] for i in range(n)]
            print({key: JSC_dict[key] for key in list(sorted_JSC.keys())[0:n]})
            print(property_training)

            df_training = pd.DataFrame()
            df_training.loc[0, "JSC"] = 1
            df_training.loc[0, f"{property}"] = 9999
            df_training.loc[0, "original_No"] = 9999
            for j in range(220):
                if molecule_new[j] > 0:
                    df_training.loc[0, df_group.iloc[j, 0]] = molecule_new[j]

            for i in range(n):
                df_training.loc[i + 1, "JSC"] = JSC_dict[list(sorted_JSC.keys())[i]]
                df_training.loc[i + 1, f"{property}"] = df.loc[list(sorted_JSC.keys())[i], f"{property}"]
                df_training.loc[i + 1, "original_No"] = list(sorted_JSC.keys())[i]
                for j in range(220):
                    if df.loc[list(sorted_JSC.keys())[i], f"Group {j + 1}"] > 0:
                        df_training.loc[i + 1, df_group.iloc[j, 0]] = df.loc[
                            list(sorted_JSC.keys())[i], f"Group {j + 1}"]

            model = ConcreteModel()
            model.I = Set(initialize=[i for i in range(220)])
            model.par = Var(model.I, within=Reals, initialize=0)
            model.par0 = Var(within=Reals, initialize=0)

            def obj_rule(model):
                error = 0
                for i in range(len(molecule_training)):
                    Tc = model.par0
                    for j in range(220):
                        Tc += model.par[j] * molecule_training[i][j]
                    error += (Tc - property_training[i]) ** 2
                error = (error / len(molecule_training)) ** 0.5
                return error

            model.obj = Objective(rule=obj_rule, sense=minimize)

            opt = SolverFactory('gams')
            io_options = dict()
            io_options['solver'] = "minos"
            io_options['mtype'] = "NLP"
            result = opt.solve(model, tee=False, keepfiles=False, io_options=io_options)

            pro_pre = np.sum([value(model.par[i]) * molecule_new[i] for i in range(220)]) + value(model.par0)

            print(value(model.par0))

            for group in df_training.columns[3:]:
                for j in range(220):
                    if df_group.iloc[j, 0] == group:
                        df_training.loc[n + 1, group] = value(model.par[j])



            return pro_pre, list(sorted_JSC.values())[0]


        df_output = df_step2_out
        for i in df_step2_out.index:
            print(i)
            molecule_new = df_step2_out.loc[i, [f"Group {j + 1}" for j in range(220)]].values
            property_pre, JSC_new = Linear2(molecule_new, molecule)
            print(property_pre, JSC_new)
            df_output.loc[i, f"{property}_pre"] = property_pre
            df_output.loc[i, "JSC"] = JSC_new
        list_df.append((df_output))
df_result=pd.concat(list_df,axis=0)
df_result=df_result.drop_duplicates().reset_index(drop=True)
df_result.to_excel(fr"{path}\result\augmentation result(with property)_all.xlsx")













