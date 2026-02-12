# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:23:07 2022

@author: Xinyu Cao

property&corresponding index:
    
0   4	tb	          0
1	4	vc	
2	4	tc	          2
3	4	pc	          3
4	2	ait	
5	4	Bcf	
6	1	Gf	
7	1	Hf	
8	4	Hfus	
9	4	Hsolp	
10	1	Hv	          4
11	4	Lc50_fm	      5
12	4	Ld50	      6
13	4	Lmv	
14	1	Logp	
15	4	Logws	
16	4	Osha_twa	
17	4	Pco	
18	4	Pka	
19	1	Tm	          1

"""


import math
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from skopt.space import Space
from skopt.sampler import Grid
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.gaussian_process import GaussianProcessRegressor


def R_square(y_test,y_predict):
    ybar = np.sum(y_test) / len (y_test)
    SSE=np.sum((y_test - y_predict)**2)
    SSR = np.sum((ybar-y_predict)**2)
    SST = np.sum((y_test - ybar)**2)
    return 1-SSE/SST
    
    
  
def isrefregirant(compoundnamelist):
    excel_name=r"C:\Users\cheese_cake\Desktop\jenny\程序\22_05WebCrawler\Refrigerants(0527 boiling&melting).xlsx"
    excel_name2=r"C:\Users\cheese_cake\Desktop\jenny\程序\22_05WebCrawler\processed(zhupengcheng).xlsx"
    df = pd.read_excel(excel_name)
    df2 = pd.read_excel(excel_name2)
    refregirant_name=list(df[df.columns[1]])
    refregirant_name2=list(df2[df2.columns[15]])
    # for i,element in enumerate(refregirant_name2):
    #     refregirant_name.append(element)
    index_of_refregirant=[]
    for i,element in enumerate(compoundnamelist):
        for j,eles in enumerate(refregirant_name):
            if not type(element)==float:
                if element.lower()==eles.lower():
                    index_of_refregirant.append(i)
    return index_of_refregirant
        
    
def GP_tuning(train_input,train_output,test_input,test_output,l):
    K_=[]
    K_trans_=[]
    if type(l)==float or type(l)==int or type(l)==np.float64:
        K_trans=my_kernal(test_input,train_input,1,l)
        K=my_kernal(train_input,train_input,1,l)
        K_.append(K)
        K_trans_.append(K_trans)
    else:
        for i in range(len(l)):
            K_trans_.append(my_kernal(test_input,train_input,1,l[i]))
            K_.append(my_kernal(train_input,train_input,1,l[i]))
    return K_,K_trans_
   
 
#普通的GP过程
def GP(train_input,train_output,test_input,test_output,l,sigma_e=1e-5,return_y=False):
    train_size=len(train_output)
    test_size=len(test_output)
    K_trans=np.zeros((test_size,train_size))
    K=np.zeros((train_size,train_size))
    if type(l)==float or type(l)==int or type(l)==np.float64:
        K_trans=K_trans+my_kernal(test_input,train_input,1,l)
        K=K+my_kernal(train_input,train_input,1,l)
    else:
        for i in range(len(l)):
            K_trans=K_trans+my_kernal(test_input,train_input,1,l[i])
            K=K+my_kernal(train_input,train_input,1,l[i])
    K2=K+sigma_e*np.eye(train_size)
    K_inv = np.linalg.lstsq(K2,np.eye(train_size),rcond=None)[0]
    y_predict=K_trans@K_inv@train_output
    rmse=RMSE(y_predict,test_output)
    if return_y:
        return rmse,y_predict
    else:
        return rmse


    

#普通的GP过程
def GP_discrete(train_input,train_output,test_input,test_output,l,sigma_e=1e-5,return_y=False):
    train_size=len(train_output)
    # print("train_size:",train_size)
    test_size=len(test_output)
    # print("test_size:",test_size)
    K_trans=np.zeros((test_size,train_size))
    # print("K_trans_size:",K_trans.shape)
    K=np.zeros((train_size,train_size))
    if type(l)==float or type(l)==int:
        K_trans=K_trans+my_kernal_discrete(test_input,train_input,1,l)
        K=K+my_kernal_discrete(train_input,train_input,1,l)
    else:
        for i in range(len(l)):
            K_trans=K_trans+my_kernal_discrete(test_input,train_input,1,l[i])
            K=K+my_kernal_discrete(train_input,train_input,1,l[i])
    K2=K+sigma_e*np.eye(train_size)
    K_inv = np.linalg.lstsq(K2,np.eye(train_size),rcond=None)[0]
    y_predict=K_trans@K_inv@train_output
    rmse=RMSE(y_predict,test_output)
    if return_y:
        return rmse,y_predict
    else:
        return rmse



def hyper_tuning(train_input,train_output,grid_number=200,fold_number=10):
    space = Space([(0.1,1),(5.0,10.0),(0.1,10.0),(10.0,100.0)]) 
    grid = Grid(border="include", use_full_layout=False)  #栅格采样
    hyperparameter = grid.generate(space.dimensions, grid_number)
    train_number=train_input.shape[0]
    validation_number=(train_number-train_number%fold_number)/fold_number
    validation_input=[]
    validation_output=[]
    train_except_validation_input=[]
    train_except_validation_output=[]
    for i in range(fold_number):
        begin_index=int(validation_number*i)
        end_index=int(validation_number*(i+1))
        delete_index=np.array([i for i in range(begin_index,end_index)])
        validation_input.append(train_input[begin_index:end_index])
        validation_output.append(train_output[begin_index:end_index])
        train_except_validation_input.append(np.delete(train_input,delete_index, axis=0))
        train_except_validation_output.append(np.delete(train_output,delete_index, axis=0))
    print("------------start hyperparameter tuning------------------")
    loss=[]     
    for i in range(grid_number):
        loss.append(validation_loss(train_except_validation_input, train_except_validation_output,
                            validation_input,validation_output,hyperparameter[i][:4],fold_number))
        print("--------------",i,"length scale:",hyperparameter[i],"average rmse on validation set:",loss[i],"--------------")
    min_loss_index=find_min_index(np.array(loss))[1]
    sigma_e=1e-5
    save={"length_scale":hyperparameter[min_loss_index],"sigma_e":sigma_e,"minimum loss":loss[min_loss_index]}
    return save
    
    
    
#超参数计算
def validation_loss(train_input,train_output,validation_input,validation_output,hyperparameter,fold_number):
    rmse_array=np.zeros(fold_number)
    for i in range(fold_number):
        rmse_vali=GP(train_input[i],train_output[i],validation_input[i],validation_output[i],hyperparameter)
        rmse_array[i]=rmse_vali
        # print("rmse for  "+str(i+1)+"th fold:",rmse_vali)
    rmse_aver=sum(rmse_array)/fold_number
    return rmse_aver


#计算误差RMSE
def RMSE(y_test,y_predict):
    y_test=y_test.reshape(-1)
    y_predict=y_predict.reshape(-1)
    rmse=math.sqrt(mean_squared_error(y_test, y_predict))
    return rmse


#利用sklearn.gaussian_process.kernels进行核计算（加速）
def my_kernal(x,y,sigma=1,l=1):
    kernel_package=sigma**2 * RBF(length_scale=l)
    return  kernel_package(x,y)


#利用sklearn.gaussian_process.kernels进行核计算（加速，离散核）
def my_kernal_discrete(x,y,sigma=1,l=1):
    x0=np.zeros_like(x)
    x1=(x0!=x).astype(int)
    x=x+x1*0.5
    y0=np.zeros_like(y)
    y1=(y0!=y).astype(int)
    y=y+y1*0.5
    kernel_package=sigma**2 * RBF(length_scale=l)
    return  kernel_package(x,y)


#filter the sample in test dataset as they are disordered
def index_of_test(train_input,train_output,data_input,data_output,dimention=424):
    train_input=train_input.reshape(-1,dimention)
    train_output=train_output.reshape(-1,1)
    data_input=data_input.reshape((-1,dimention))
    data_output=data_output.reshape(-1,1)
    index_exist=np.zeros(data_output.shape[0])
    for i in range(data_output.shape[0]):
        for j in range (train_output.shape[0]):
            if (data_output[i]==train_output[j]).all() and (data_input[i]==train_input[j]).all():
               index_exist[i]=1
    index_test=[]
    for i in range(len(index_exist)):
        if index_exist[i]==0:
            index_test.append(i)
    return index_test

def index_of_train(train_input,train_output,data_input_atom,data_output_atom,dimention=424):
    train_input=train_input.reshape(-1,dimention)
    train_output=train_output.reshape(-1,1)
    data_input_atom=data_input_atom.reshape((-1,dimention+15))
    data_output_atom=data_output_atom.reshape(-1,1)
    index_exist=np.zeros(data_output_atom.shape[0])
    index_train=np.zeros(train_output.shape[0])
    for j in range(train_output.shape[0]):
        for i in range(data_output_atom.shape[0]):
            if (data_input_atom[i,15:]==train_input[j]).all(): #(data_output_atom[i]==train_output[j]).all() and 
               index_exist[i]=1
               index_train[j]=i
    return index_train

#filter the same input  and return the index
def index_analysis(data_input,data_output,delete_return=False):
    already_find=[]
    data_output=data_output.reshape(-1,1)
    same_input=[]
    delete_index=[i for i in range(len(data_output))]
    for i in range(data_output.shape[0]):
        for j in range (data_output.shape[0]):
            if (data_input[i]==data_input[j]).all() and i!=j and (not(j in already_find)):
               same_input.append((i,j))
               already_find.append(j)
        already_find.append(i)
    #Divide them into different groups and record the output
    group=[]
    same_index_group=[]
    before_index0=same_input[0][0]
    for i,element in enumerate(same_input):   
        if element[0]==before_index0:
            if i==0:
                group.append([element[0],data_output[element[0]]])
                group.append([element[1],data_output[element[1]]])
            else:   
                group.append([element[1],data_output[element[1]]])
            if i==len(same_input)-1:
                group.append(data_input[before_index0])
                same_index_group.append(group)
        else:
            group.append(data_input[before_index0])
            same_index_group.append(group)
            group=[]
            group.append([element[0],data_output[element[0]]])
            group.append([element[1],data_output[element[1]]])
            if i==len(same_input)-1:
                group.append(data_input[before_index0])
                same_index_group.append(group)
        before_index0=element[0]     
    #Select which one to delete
    for i,element in enumerate(same_index_group):
        number=len(element)-1
        if number==2:
            delete_index.remove(element[1][0])
            # print(element[1][0])
        else:
            aver=0
            index_record=[]
            for j in range(number):
                aver=(aver*j+element[j][1])/(j+1)
            for j in range(number):
                index_record.append([element[j][0],abs(element[j][1]-aver)])
            compare_index=index_record[0][0]
            compare_value=index_record[0][1]
            for j in range(1,number):
                if index_record[j][1]>compare_value:
                    delete_index.remove(index_record[j][0])
                    # print(index_record[j][0])
                else:
                    delete_index.remove(compare_index)
                    # print(compare_index)
                    compare_index=index_record[j][0]
                    compare_value=index_record[j][1]
    if delete_return:
        return same_index_group,delete_index
    else:
        return same_index_group

#随意划分训练集
def data_processing2(data_input,data_output,percentage=0.95):
    same_index_group,delete_index=index_analysis(data_input,data_output,1)
    data_input=data_input[delete_index]
    data_output=data_output[delete_index]
    data_num=len(data_output)
    train_num=int(len(data_output)*percentage)
    sample = range(data_num)
    train_index=random.sample(sample,train_num)
    train_input=data_input[train_index]
    train_output=data_output[train_index]
    test_output=np.delete(data_output, train_index, axis=None)
    test_input=np.delete(data_input, train_index, axis=0)
    return data_input,data_output,train_input,train_output,test_input,test_output
    
    
    
#data processing 
def data_processing(data_input,data_output,train_input,train_output):
    same_index_group,delete_index=index_analysis(data_input,data_output,1)
    data_input=data_input[delete_index]
    data_output=data_output[delete_index]
    same_input_index,delete_index=index_analysis(train_input,train_output,1)
    train_input=train_input[delete_index]
    train_output=train_output[delete_index]
    deal_with_train(train_input,train_output,data_input,data_output)
    data_input=data_input.astype(np.float64)
    train_input=train_input.astype(np.float64)
    index_test=index_of_test(train_input, train_output, data_input, data_output)
    test_input=data_input[index_test]
    test_output=data_output[index_test]
    return data_input,data_output,train_input,train_output,test_input,test_output


#data processing 
def data_processing3(data_input,data_output,train_input,train_output):
    same_index_group,delete_index=index_analysis(data_input,data_output,1)
    data_input=data_input[delete_index]
    data_output=data_output[delete_index]
    same_input_index,delete_index=index_analysis(train_input,train_output,1)
    train_input=train_input[delete_index]
    train_output=train_output[delete_index]
    deal_with_train(train_input,train_output,data_input[:,15:],data_output)
    data_input=data_input.astype(np.float64)
    train_index=index_of_train(train_input, train_output, data_input, data_output).astype(int)
    train_input=data_input[train_index]
    index_test=index_of_test(train_input, train_output, data_input, data_output,439)
    test_input=data_input[index_test]
    test_output=data_output[index_test]
    return data_input,data_output,train_input,train_output,test_input,test_output

#multiply different coefficients in different dimensions
def input_modi(coef,alfa,dataset):
    dataset_modi=np.zeros_like(dataset)
    dataset=dataset.reshape(-1,dataset.shape[1])
    for i1,element in enumerate(dataset.T):  
        if coef[i1]!=0:
            dataset_modi[:,i1]=element*((np.abs(coef[i1]))**alfa) 
    return dataset_modi


#analyse the percentage of data within 1%,5%,10%
def error_analysis(y_predict,y_true):
    number_sample=len(y_predict)
    within_one_percent=np.zeros(number_sample)
    within_five_percent=np.zeros(number_sample)
    within_ten_percent=np.zeros(number_sample)
    relative_error=np.zeros(number_sample)
    for i in range(number_sample):
        relative_error[i]=abs(y_predict[i]-y_true[i])/y_true[i]
        if relative_error[i]<=0.01:
            within_one_percent[i]=1
        if relative_error[i]<=0.05:
            within_five_percent[i]=1
        if relative_error[i]<0.1:
            within_ten_percent[i]=1
    error_one_percent=sum(within_one_percent)/number_sample
    error_five_percent=sum(within_five_percent)/number_sample
    error_ten_percent=sum(within_ten_percent)/number_sample
    error=[error_one_percent,error_five_percent,error_ten_percent]
    error=np.array(error).reshape(1,3)
    return {"relative_error":relative_error,"error_one_percent":error_one_percent,"error_five_percent":error_five_percent,"error_ten_percent":error_ten_percent,"error":error}



def mll_print(K,y_train):
    if np.linalg.det(K)<1e-4:
        K=np.identity(K.shape[0])*1e-1+K
    n=K.shape[0]    
    mll1=float(0.5*np.matmul(np.matmul((y_train.reshape(1,n)),np.linalg.pinv(K)),y_train.reshape(n,1)))
    mll2=0.5*math.log(np.linalg.det(K))
    mll3=0.5*n*math.log(2*math.pi)
    mll=-mll1-mll2-mll3
    # print("mll:",mll)
    return -mll



#make the output in the test dataset same as that in the training dataset
def deal_with_train(train_input,train_output,data_input,data_output):
    for i,ele in enumerate(train_input):
        for j,eles in enumerate(data_input):
            if (ele==eles).all():
                train_output[i]=data_output[j]
                # print("deal---",i,j)
                

#Probably useless, but can't bear to delete it
def feature_extraction(coef,data_input):
    index1=[]
    index2=[]
    index3=[]
    index4=[]
    count=0
    for i,element in enumerate(coef):
        # print(i,element)
        if element<=0.25:
            count+=1
            index1.append(i)
        if 0.25<element<=0.5:
            count+=1
            index2.append(i)
        if 0.5<element<=0.75:
            count+=1
            index3.append(i)
        if element>0.75:
            count+=1
            index4.append(i)
    if count==len(coef):
        group1,group2,group3,group4=data_input[:,index1],data_input[:,index2],data_input[:,index3],data_input[:,index4]
        group1,group2,group3,group4=group1.sum(axis=1),group2.sum(axis=1),group3.sum(axis=1),group4.sum(axis=1)
        data_input_fs=np.vstack((group1,group2,group3,group4)).T
        return data_input_fs

    
#find minimum value and its coefficient in a two-dimentional matrix    
def find_min_index(x):
    if len(x.shape)==2:
        index=[0,0]
        min_value=x.flatten()[0]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # print(i,j)
                if x[i,j]<min_value:
                    index[0]=i
                    index[1]=j
                    min_value=x[i,j]
    if len(x.shape)==1:
        index=0
        min_value=x[0]
        for i in range(len(x)):
            if x[i]<min_value:
                index=i
                min_value=x[i]      
    return min_value,index
    
 
    

            


def plot_quartile(data):
    data = pd.DataFrame(data)
    plt.rcParams['savefig.dpi'] = 1000 # 图片像素
    plt.show()


    
def input_modi_distort(coef,dataset):
    dataset_modi=np.zeros_like(dataset)
    dataset=dataset.reshape(-1,dataset.shape[1])
    for i1,element in enumerate(dataset.T):  
        if coef[i1]!=0:
            dataset_modi[:,i1]=np.log(element+1)/np.log(1/(coef[i1]+1))
    return dataset_modi
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    