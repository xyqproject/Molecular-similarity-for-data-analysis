import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
import pickle
import warnings
from skopt.space import Space
from skopt.sampler import Grid
from ProductDesignFunction import RMSE,index_of_train, index_of_test, GP_tuning, RMSE, input_modi, error_analysis, \
    validation_loss, find_min_index, GP, plot_quartile, hyper_tuning
warnings.filterwarnings("ignore")


def fun_wholemodel(train_input,train_output,test_input,test_output):
    # Linear prior...............................
    print("-----------------simple1----------------")
    svr_lin = SVR(kernel="linear", C=100, gamma="auto", epsilon=0.4)
    model_simple = svr_lin.fit(train_input, train_output)
    print("-----------------stop training------------------")
    test_predict_simple = model_simple.predict(test_input)
    rmse_test_simple = RMSE(test_predict_simple, test_output)
    print("rmse_test_simple:", rmse_test_simple)
    train_predict_simple = model_simple.predict(train_input)
    rmse_train_simple = RMSE(train_predict_simple, train_output)
    print("rmse_train_simple:", rmse_train_simple)

    # GP model...............................
    fold_number = 5
    alpha_num = 5
    grid_number = 100

    test_output_linear = test_predict_simple
    test_output_nonlinear = test_output - test_output_linear
    train_output_linear = train_predict_simple
    train_output_nonlinear = train_output - train_output_linear

    space = Space([(0.1, 1), (5.0, 10.0), (0.1, 10.0), (10.0, 100.0)])
    grid = Grid(border="include", use_full_layout=False)  # 栅格采样
    hyperparameter = grid.generate(space.dimensions, grid_number)
    alfaset = np.linspace(1.2, 4, alpha_num)

    #start to train GP.........................
    loss_record = []
    l_record = []
    for i2 in range(len(alfaset)):
        alfa = alfaset[i2]
        print(f"--------------alfa:{alfa}----------------")
        l = np.ones(len(hyperparameter[0])) * 10
        test_input_distort = np.log(test_input + 1) / np.log(alfa)
        train_input_distort = np.log(train_input + 1) / np.log(alfa)
        sigma_e = 1e-5
        hyper = hyper_tuning(train_input_distort, train_output_nonlinear, grid_number, fold_number)
        l = hyper['length_scale']
        l_record.append(l)
        loss_record.append(hyper["minimum loss"])
    index_ = find_min_index(np.array(loss_record))[1]
    l = l_record[index_]
    alfa = alfaset[index_]

    #prediction..................................
    test_input_distort = np.log(test_input + 1) / np.log(alfa)
    train_input_distort = np.log(train_input + 1) / np.log(alfa)
    test_predict =GP(train_input_distort, train_output_nonlinear, test_input_distort, test_output_nonlinear, l, 1e-5, 1)[1]
    rmse_test = RMSE(test_output, test_predict + test_output_linear)
    print("rmse_modi_test:", rmse_test)
    train_predict =GP(train_input_distort, train_output_nonlinear, train_input_distort, train_output_nonlinear, l, 1e-5, 1)[1]
    rmse_train = RMSE(train_output, train_predict + train_output_linear)
    print("rmse_modi_train:", rmse_train)

    save = {"length_scale": l, "sigma_e": 1e-5, "alfa": alfa,
            "test_predict": test_predict + test_output_linear,
            "train_predict": train_predict + train_output_linear, }
    return train_predict + train_output_linear,test_predict + test_output_linear


path=r"\adaptive modeling and reliability assessment"
# similarity model.....................................
property_list=["vc", "pc", "ait", "gf", "hf", "hsolp","hv", "lmv","tc"]

for property in property_list:
    df = pd.read_excel(fr"{path}\dataset\{property}.xlsx", index_col=0)
    train_num = max(int(0.8 * len(df.index)),len(df.index)-200)
    molecule=[df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(1,train_num+1)]

    n = 50
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
        _,pro_pre=fun_wholemodel(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array([molecule_new]).astype(float),np.array([0]).astype(float))
        return pro_pre[0],list(sorted_JSC.values())[0]

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
    df_output.to_excel(fr"{path}\result\GPR_{property}_output_{n}.xlsx")


