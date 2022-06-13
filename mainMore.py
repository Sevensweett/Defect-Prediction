#_*_conding=utf-8_*_
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.colors
import matplotlib.pyplot as plt
from functools import reduce

from scipy import stats
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances_argmin
import math
from scipy.integrate import tplquad,dblquad,quad
import scipy.stats

#from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD 

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


from scipy.interpolate import make_interp_spline

from sklearn import neighbors

from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import GaussianNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score,fbeta_score
from sklearn.metrics import roc_auc_score
import numpy as np;
import matplotlib.pyplot as plt;


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN

from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import random

from sklearn import preprocessing
from sklearn.metrics import accuracy_score


from collections import  Counter

# 绘制背景的边界
from matplotlib.colors import ListedColormap

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 数据预处理 usecols=np.arange(0,50)
def dataGetPromise12():
    data = []
    data1 = np.loadtxt('..\\所用数据集\\promise\\ant-1.7.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data2 = np.loadtxt('..\\所用数据集\\promise\\camel-1.6.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data3 = np.loadtxt('..\\所用数据集\\promise\\ivy-2.0.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data4 = np.loadtxt('..\\所用数据集\\promise\\jedit-4.0.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data5 = np.loadtxt('..\\所用数据集\\promise\\log4j-1.0.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data6 = np.loadtxt('..\\所用数据集\\promise\\lucene-2.4.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data7 = np.loadtxt('..\\所用数据集\\promise\\poi-3.0.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data8 = np.loadtxt('..\\所用数据集\\promise\\synapse-1.2.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data9 = np.loadtxt('..\\所用数据集\\promise\\tomcat.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data10 = np.loadtxt('..\\所用数据集\\promise\\velocity-1.6.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data11 = np.loadtxt('..\\所用数据集\\promise\\xalan-2.4.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data12 = np.loadtxt('..\\所用数据集\\promise\\xerces-1.3.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(3,24))
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    data.append(data7)
    data.append(data8)
    data.append(data9)
    data.append(data10)
    data.append(data11)
    data.append(data12)
    return data

def dataGetNasa7():
    data = [];
    data1 = np.loadtxt('..\\所用数据集\\nasa\\CM1.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data2 = np.loadtxt('..\\所用数据集\\nasa\\KC1.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data3 = np.loadtxt('..\\所用数据集\\nasa\\KC3.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data4 = np.loadtxt('..\\所用数据集\\nasa\\MC2.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data5 = np.loadtxt('..\\所用数据集\\nasa\\MW1.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data6 = np.loadtxt('..\\所用数据集\\nasa\\PC2.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data7 = np.loadtxt('..\\所用数据集\\nasa\\PC4.csv', dtype=np.float64,delimiter=',', skiprows=1,usecols=np.arange(0,21))
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    data.append(data7)
    return data

def dataPreTreated(data):
    List = [];
    target = [];
    for i in range(len(data)):
        data_new, data_t = np.split(data[i], [-1, ], axis=1)
        List.append(preprocessing.scale(data_new))
        for j in range(len(data_t)):
            if(data_t[j] > 1):
                data_t[j] = 1
        target.append(data_t)
    return List, target


import GMM_kp_sample
#data = dataGetPromise12()
data = dataGetNasa7()
data, target = dataPreTreated(data)
print(len(data))
print(data[0].shape)
print(data[0][0])

from sklearn.metrics import classification_report
from liblinear.liblinearutil import *
# from liblinearutil import *
from sklearn.metrics import confusion_matrix
k_p = [0.25, 0.5, 1, 2, 4]
k_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#JS = np.loadtxt('..\\js-结果\\promise12-NEW.txt', dtype=np.float,delimiter=',');
JS = np.loadtxt('..\\js-结果\\nasa7-NEW.txt', dtype=np.float, delimiter=',')

size = len(data)


result_f_binary = np.zeros((size, size))
result_auc = np.zeros((size, size))
result_g_mean = np.zeros((size, size))

for numi in range(0,size):
    argsort = np.argsort(JS[numi])

    # 储存最好结果下的y值
    all_y = np.zeros((size, len(data[numi])))

    data_test = data[numi]
    target_test = target[numi]
    for kpi in range(0, size):
        # 确定当前训练项目和目标项目
        data_jschoose = data[kpi]
        target_jschoose = target[kpi]
        
        auc_max = 0
        y_max = np.zeros(len(data[numi]))
        # 计算出kp各个结果的值
        for k_i in range(5):
            for percent_i in range(9):
                # print("k",k_i,"per",percent_i)
                # 使用k-percent方法
                ros = GMM_kp_sample.GMM_Separate(k=k_p[k_i], percent=k_percent[percent_i])
                X_resampled, y_resampled = ros.fit_sample(data_jschoose, target_jschoose)
                
                y_resampled_count = sum(y_resampled == 1)
                y_resampled_count_0 = sum(y_resampled == 0)

                #某一项太少了不执行此项
                if(y_resampled_count <= 1) | (y_resampled_count_0 <= 1):
                    f_binary = 0
                    auc = 0
                    g_mean = 0
                else:
                    clf = train(y_resampled.ravel(), X_resampled, '-s 0')
                    p_label, p_acc, p_val = predict(target_test.ravel(), data_test, clf)

                    # mnb = KNeighborsClassifier(n_neighbors=5);
                    # mnb = GaussianNB();  # 使用默认配置初始化朴素贝叶斯
                    # mnb.fit(X_resampled,y_resampled.ravel())
                    # p_label = mnb.predict(np.array(data_test));

                    auc = roc_auc_score(target_test, p_label)

                    if(auc > auc_max):
                        auc_max = auc
                        y_max = p_label
                        
        # 保存结果最好的y值
        all_y[kpi] = y_max

            
    for k in range(3, size):
        sum_k = 0
        per_k = np.zeros(size)
        # 当此时挑选ki个项目时
        for ki in range(k):
            per = 1/JS[numi][argsort[ki+1]]
            sum_k = sum_k + per
            per_k[ki] = per
            

        # 获取当前占比
        per_k = per_k/sum_k

        y_result = np.zeros(len(data_test))
        # 针对当前选择的项目进行建模,argsort[ki+1]当前选择项目序号
        for ki in range(k):
            # 预测总值为各部分占比和相加
            y_result = y_result + per_k[ki] * all_y[argsort[ki+1]]

        for y_i in range(len(data_test)):
            if(y_result[y_i] >= 0.5):
                y_result[y_i] = 1
            else:
                y_result[y_i] = 0

        f_binary = f1_score(target_test, y_result, average="binary")
        auc = roc_auc_score(target_test, y_result)
        C_matrix = confusion_matrix(target_test, y_result, labels=[1, 0])
        tp = C_matrix[0][0]
        tn = C_matrix[1][1]
        fp = C_matrix[1][0]
        fn = C_matrix[0][1]
        re = tp/(tp+fn)
        pf_1 = tn/(tn + fp)
        g_mean = math.sqrt( pf_1 * re )

  
        # 将结果保存下来
        result_f_binary[numi][k] = f_binary
        result_auc[numi][k] = auc
        result_g_mean[numi][k] = g_mean
        
    
result_f_binary = np.array(result_f_binary)
result_auc = np.array(result_auc)
result_g_mean = np.array(result_g_mean)

print("----------list---------------")
print("f_binary")
print(result_f_binary)
print("auc")
print(result_auc)
print("g_mean")
print(result_g_mean) 

print("f_binary")
for var_f_binary in result_f_binary.tolist():
    print(var_f_binary)
print("auc")
for var_auc in result_auc.tolist():
    print(var_auc)
print("g_mean")
for var_g_mean in result_g_mean.tolist():
    print(var_g_mean) 
