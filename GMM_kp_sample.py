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

from imblearn.over_sampling import RandomOverSampler

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
import numpy as np  
import matplotlib.pyplot as plt 


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

class GMM_Separate():
    def __init__(self, k=1,percent=0.1):
        self.k = k;
        self.percent = 1-percent;

    def fit_sample(self,data,target):
        length = data.shape[0];

        K = math.ceil(length ** 0.5 * self.k)
        dis_knn = np.zeros(length);


        threshold = dis_knn[sort[int(sort.shape[0] * self.percent)]];

        #获取相对密度较大值
        for i in range(int(sort.shape[0])):
            if(dis_knn[i] >= threshold):
                circle_data.append(data[i])
                circle_target.append(target[i])

        circle_data = np.array(circle_data)
        circle_target = np.array(circle_target)
        return circle_data,circle_target;
         
