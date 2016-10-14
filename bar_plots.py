import scipy.stats.distributions as dist
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

def get_fractional_errors(k,n,c=0.683):
    
    p_lower = dist.beta.ppf((1-c)/2.,k+1,n-k+1)
    p_upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
    
    return p_lower,p_upper


def get_tables_with_errors(table,normalise='fraction',error=True,column_totals=None):
    
    '''
    --- Method for calculating the 'counting' errors for individual columns
    of a table---
    
    Inputs:
    -------
    table: set of values to input.
    
    normalise: if 'fraction', all values add to 1. If 'percent' all values add to 100.
    Else, raw values will be used.
    
    error: if True, then the tables with low/high fractions will be the errors;
    if False, then they will correspond to fractions. (eg. 0.75+/-0.05 will return 
    0.7 and 0.8 if False, or 0.05 and 0.05 if True).
    
    column_totals: if None, then the column total is the sum of that column. 
    
    Outputs:
    --------
    fracs: table of fractions/ numbers (with same shape as table. If normalise isn't 
    'percent' or 'fraction',fracs will be the same as Table)
    
    fracs_low, fracs_high: exactly the same format as fracs/table, but with corresponding
    error values.
    '''
    
    fracs_low = Table()
    fracs = Table()
    fracs_high = Table()
    
    for i,c in enumerate(table.colnames):
        column = table[c]
        if column_totals == None:
            column_total = np.sum(column)
        else:
            column_total = column_totals[i]
        f = column/column_total
        f_low, f_high = get_fractional_errors(column,column_total)
        if error:
            f_low = f-f_low
            f_high = f_high-f
        if normalise == 'percent':
            f, f_low, f_high = [f*100,f_low*100,f_high*100]
        elif normalise == 'fraction':
            f, f_low, f_high = [f,f_low,f_high]
        else:
            f, f_low, f_high = [f*column_total,f_low*column_total,f_high*column_total]
        
        fracs_low[c] = f_low
        fracs[c] = f
        fracs_high[c] = f_high
    
    return fracs, fracs_low, fracs_high
    

def comparison_bar_chart(table,labels,colors,normalise=None,width=0.5,alpha=1
                         ,linewidth=3,linecolor='k',ylabel='N',column_totals=None,right_space=2):
    
    '''
    --- Compare number of objects that fit into certain categories. 
    (eg. detected vs. non-detected) ---
    
    Inputs:
    -------
    table: astropy table, with number of columns=number of categories to compare (eg. arm number).
    
    labels: what each of the rows correspond to (length=N_rows). (eg. ['detected','not detected'])
    
    colors: colors for each of the bars
    
    normalise: if 'fraction', all values add to 1. If 'percent' all values add to 100.
    Else, raw values will be used.
    
    width: width of each group.
    
    alpha, linewidth,linecolor: bar formatting.
    
    ylabel: label for the y-axis.
    
    column_totals: List of totals for each of the coluns (length = N_columns). 
    If None, then the column total is the sum of that column.
    
    right_space: space left on the right hand side of the plot, for fitting the legend.
    
    Outputs:
    --------
    fracs: table of fractions/ numbers (with same shape as table. If normalise isn't 
    'percent' or 'fraction',fracs will be the same as Table)
    
    fracs_low, fracs_high: exactly the same format as fracs/table, but with corresponding
    error values.
    '''
    
    column_labels = table.colnames
    N_columns = len(column_labels) # 'columns' = number of bar 'groups'
    N_class = len(table) # 'class' = number of 'thin' bars.
    
    column_centres = [j+0.5 for j in range(N_columns)]
    class_width = width/N_class
    bar_edges = np.linspace(-width/2,width/2,N_class+1)
    bar_centres = [bar_edges[j]+(bar_edges[j+1]-bar_edges[j])/2 for j in range(N_class)]
    bar_lefts = [b-class_width/2 for b in bar_centres]
    
    fracs, fracs_low, fracs_high = get_tables_with_errors(table,normalise
                                                          ,error=True,column_totals=column_totals)
    
    for n in range(N_class):
        offset_left, offset_centre = [bar_lefts[n],bar_centres[n]]
        class_lefts = [j+offset_left for j in column_centres]
        class_centres = [j+offset_centre for j in column_centres]
        class_f = [fracs[c][n] for c in column_labels]
        class_low_error = [fracs_low[c][n] for c in column_labels]
        class_high_error = [fracs_high[c][n] for c in column_labels]
        
        plt.bar(class_lefts,class_f,width=class_width,label=labels[n]
                ,alpha=alpha,color=colors[n],lw=linewidth,edgecolor=linecolor,linewidth=linewidth)

        _ = plt.errorbar(class_centres,class_f, yerr=[class_low_error,class_high_error]
                         ,markersize=None,ecolor=linecolor,capsize=4,elinewidth=linewidth
                         ,linewidth=0,capthick=linewidth)
    plt.legend(fontsize=15)
    plt.xlim(column_centres[0]-1,column_centres[-1]+right_space)
    plt.xticks(column_centres,column_labels)
    plt.ylabel(ylabel)
    
    return fracs, fracs_low, fracs_high