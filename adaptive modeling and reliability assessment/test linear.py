import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pyomo.opt import SolverFactory
from pyomo.environ import *

#...................................................................linear model....................................................
path=r"\adaptive modeling and reliability assessment"
property_list=["vc", "pc", "ait", "gf", "hf", "hsolp","hv", "lmv","tc"]

for property in property_list:
    df = pd.read_excel(fr"{path}\dataset\{property}.xlsx",index_col=0)
    train_num = max(int(0.8 * len(df.index)),len(df.index)-200)
    molecule=[df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(1,train_num+1)]
    n=50

    def JSC(list1,list2):     #index3+index2
        intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
        union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
        res=(np.product(intersection)-1)/(np.product(union)-1)
        return res

    def Linear(molecule_new,molecule):
        JSC_dict={i+1:JSC(molecule_new,molecule[i]) for i in range(len(molecule))}
        sorted_JSC = {key: JSC_dict[key] for key in sorted(JSC_dict.keys(), key=JSC_dict.get,  reverse=True)}
        molecule_training=[df.loc[list(sorted_JSC.keys())[i], [f"Group {j + 1}" for j in range(424)]].values for i in range(n)]
        property_training=[df.loc[list(sorted_JSC.keys())[i], f"{property}"] for i in range(n)]

        model = ConcreteModel()
        model.I = Set(initialize=[i for i in range(424)])
        model.par = Var(model.I, within=Reals, initialize=0)
        model.par0 = Var(within=Reals, initialize=0)

        def obj_rule(model):
            error = 0
            for i in range(len(molecule_training)):
                Tc = model.par0
                for j in range(424):
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

        pro_pre=np.sum([value(model.par[i])*molecule_new[i] for i in range(424)])+value(model.par0)
        return pro_pre,list(sorted_JSC.values())[0]



    df_output=pd.DataFrame()
    for i in range(train_num+1,len(df.index)+1):
        print(i)
        molecule_new = df.loc[i, [f"Group {j + 1}" for j in range(424)]].values
        property_new=df.loc[i, f"{property}"]
        property_pre,JSC_new=Linear(molecule_new, molecule)
        print(property_pre,property_new,JSC_new)
        df_output.loc[i,f"{property}_pre"]=property_pre
        df_output.loc[i, f"{property}_real"] = property_new
        df_output.loc[i, "JSC"] = JSC_new
        df_output.loc[i, "error"] = abs((property_pre-property_new)/property_new)
    df_output.to_excel(fr"{path}\result\SWR_{property}_output_{n}.xlsx")







