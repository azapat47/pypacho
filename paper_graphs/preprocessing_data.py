import pandas as pd
import numpy as np

df = pd.read_csv('out_op_float_unitary.csv', sep = ',', index_col=0)

cuda=df[df['platform'] == 'cuda']
numpy=df[df['platform'] == 'numpy']
cl=df[df['platform'] == 'cl']

platforms = [cuda,cl,numpy]

## Operations Lab
group_by = 'size_a_m'
methods = ['suma','resta','punto matriz x matriz','norma','punto vector x vector', 'transpuesta','punto matriz x vector', 'multiplicacion', 'division']

## Methods Lab 
#group_by = 'size'
#methods = ['GD', 'CG', 'jacobi']

for p in platforms:
    for i in methods:
        s = p[p.method==i].groupby(group_by).mean()
        outdf = pd.DataFrame(s['time'])
        outdf.index.rename('size', inplace=True)
        strPlaform = str(p.platform.iloc[0])
        outdf.rename(columns={'time':strPlaform}, inplace=True)
        if(not outdf.empty):
           filename = strPlaform+'_'+i+'.data'
           outdf.to_csv(filename)
        
