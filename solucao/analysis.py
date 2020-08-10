import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

def data_investigation(data):
    print('Number of instances : ' + str(data.shape[0]))
    print('Number of variables : ' + str(data.shape[1]))
    print('-'*20)
    print('Attributes, data type and ratio of unique instances por total non-null:')
    for i in range(data.shape[1]):
        print('\t - ' + str(data.columns[i]) + ', ' + str(data.dtypes[i]) 
              + ', ' + str(len(data[data.columns[i]].value_counts())) + '/' + 
             str(sum(data[data.columns[i]].value_counts())))
    
    print('-'*20)
    print('Attributes that have missing values: ')
    sum_missing_val = data.isnull().sum()
    print(sum_missing_val[sum_missing_val>0])
    print('-'*20)
    print('Pictorial representation of missing values:')
    plt.figure(figsize=(10,8))
    sns.heatmap(data.isnull(), yticklabels = False, cmap = 'gray')
    plt.show()
    
def corr_matrix_plot(corr_matrix):
    plt.figure(figsize=(10,8))
    
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot = True)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
        horizontalalignment='right')
    
def hist_plot(df, target = False):
    if target == False:
        target = df.columns
        
    J, L = get_subplot_dims(target)
    
    plt.figure(figsize = (12, 12))
    for i in range(len(target)):
        plt.subplot(J, L, i+1)
        sns.countplot(x = target[i], data = df)
        plt.xticks(rotation=60)
        
    plt.tight_layout()
    
    plt.show()
    
def get_subplot_dims(targets):
    a = (int(len(targets)/3))*3
    
    if a == len(targets):
        j = int(a/3)
    elif a < len(targets):
        j = int(a/3) + 1
    return(j, 3)   

def stack_plot(df, target, stack_target, colors = ['black', 'red', 'yellow']):
    stack_uniques = df[stack_target].unique()
    plt.figure(figsize = (12, 12))
    m = 0
    a = [0, 0, 0]
    for j in stack_uniques:
        data_sub = df[df[stack_target].isin(stack_uniques[0:len(stack_uniques)-m])]
        a[m] = data_sub
        m += 1
    fig, ax = plt.subplots()
    for k in range(3):
        sns.countplot(x = target, data = a[k], color = colors[k], ax = ax)

    plt.xticks(rotation=60)
        
    plt.tight_layout()
    
    plt.show()

def hist_stack_plot(df, target, stack_target):
    J, L = get_subplot_dims(target)
    
    plt.figure(figsize = (12, 12))
    for i in range(len(target)):
        plt.subplot(J, L, i+1)
        sns.countplot(x = target[i], data = df, hue = stack_target)
        plt.xticks(rotation=60)
        
    plt.tight_layout()
    plt.show()

    
def scatter_plot_num(df, target, stack_target, alpha=1, title=' '):
    L = len(target)
    L2 = L**2
    z = 1
    legend = False
    
    plt.figure(figsize = (15, 15))
    for i in range(L):
        for j in range(L):
            plt.subplot(L, L, z)
            z += 1
            if z == L2+1:
                legend = 'brief'
            sns.scatterplot(data = df, x = target[j], y = target[i], hue = stack_target, legend = legend, alpha = alpha, palette='colorblind')
    
    plt.title(title)
    plt.show()
    
def rel_plot_num(df, target, hue, title = ' '):
    L = len(target)
    L2 = L**2
    z = 0
    legend = False
    fig, ax = plt.subplots(L, L, figsize = (15, 15))
    for i in range(L):
        for j in range(L):
            if z == L2-1:
                legend = 'brief'
            sns.lineplot(data = df, x = target[j], y = target[i], hue = hue, legend = legend, ax = ax[i, j])
            z += 1
    plt.title(title)
    plt.show()
    
def scatter_sep_plot_num(data, targets, sep, alpha = 0.3):
    g = sns.FacetGrid(data, col = sep)
    g.map(sns.scatterplot, targets[0], targets[1], alpha = alpha)
    g.add_legend()
    plt.show()
        
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def corr_matrix_v(data, targets):
    L = len(targets)
    L_range = range(L)
    corr = pd.DataFrame(0, index=targets, columns=targets)
    
    for i in L_range:
        for j in L_range:
            v = cramers_v(data[targets[i]], data[targets[j]])
            corr.iloc[i, j] = v
    
    return (corr).round(4)

def eta_coef(data, num, cat):
    cat_uniques = data[cat].unique()
    
    num_vals = data[num]
    mean = num_vals.mean()
    SSt = ((num_vals-mean)**2).sum()
    SSe = 0
    
    for i in cat_uniques:
        num_x = num_vals[data[cat]==i]
        n_x = num_x.shape[0]
        mean_x = num_x.mean()
        val_x = n_x * (mean_x-mean)**2
        SSe = SSe + val_x
    
    eta2 = SSe/SSt
    return round(np.sqrt(eta2), 4)

def corr_matrix_eta(data, num_targets, cat):
    corr = pd.DataFrame(0, columns = num_targets, index = [cat])
    
    L = len(num_targets)
    
    for i in range(L):
        eta_val = eta_coef(data, num_targets[i], cat)
        corr.iloc[0, i] = eta_val
    
    return corr.round(4)
    
def cut_num(df, targets, bins, y):
    df_cut = pd.DataFrame(index = df.index, columns = targets)
    df_cut[y] = df[y]
    for i in range(len(targets)):
        df_cut[targets[i]] = pd.cut(df[targets[i]], bins[i])
    
    return df_cut