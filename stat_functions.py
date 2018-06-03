import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# obtain data
def get_data(filename):
    data = np.loadtxt(filename)
    return data

def get_data_many(filename):
    data_raw = np.loadtxt(filename)
    data = data_raw.transpose()
    #print(data)
    return data

# describing data

def describe_data(data):
    """ data is a numpy array of values"""
    min_ = np.amin(data)
    max_ = np.amax(data)
    mean_ = np.mean(data)
    median_ = np.median(data)
    mode_ = st.mode(data)
    std_ = np.std(data)
    var_ = np.var(data)
    skew_ = st.skew(data)
    kurtosis_ = st.kurtosis(data)
    q_25, q_50, q_75 = np.percentile(data, [25,50,75])
    basic = 'Min: %s\nMax: %s\nMean: %s\nMedian: %s\nMode: %s\nVar: %s\nStd: %s'
    other = '\nSkew: %s\nKurtosis: %s\nQ25: %s\nQ50: %s\nQ75: %s'
    all_ = basic + other
    print(all_ % (min_,max_,mean_,median_,mode_,var_,std_,skew_,kurtosis_,q_25,q_50,q_75))
    return (min_,max_,mean_,median_,mode_,var_,std_,skew_,kurtosis_,q_25,q_50,q_75)

# visualizing data
def histogram(data,title,xlabel,ylabel,bins=25):
    plt.hist(data,bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def histogram_norm(data,title,xlabel,ylabel,bins=20):
    plt.hist(data,normed=1,bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    min_,max_,mean_,median_,mode_,var_,std_,*X = describe_data(data)
    x = np.linspace(min_,max_,1000)
    pdf = st.norm.pdf(x,mean_,std_)
    plt.plot(x,pdf,'r')    
    plt.show()

def box_plot(data, labels):
    plt.boxplot(data,labels=labels)
    plt.show()

def test_normal(data,opt):
    config1 = data[0]
    config2 = data[1]
    config3 = data[2]
    config4 = data[3]
    for values in [config1,config2,config3,config4]:
        if(opt==1):
            print(test_normal_ks(values))
            print()
        else:
            print(test_normal_sw(values))
            print()

def test_normal_ks(data):
    """Kolgomorov-Smirnov"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.kstest(norm_data,'norm')

def test_normal_sw(data):
    """Shapiro-Wilk"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.shapiro(norm_data)

# Non Parametric

def wilcoxon(data1,data2):
    """
    non parametric
    two samples
    dependent
    """     
    return st.wilcoxon(data1,data2)

def friedman_chi(data):
    """
    non parametric
    many samples
    dependent
    """     
    F,pval = st.friedmanchisquare(*data)
    return (F,pval)    
       
