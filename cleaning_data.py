import os 
import numpy as np
import pandas as pd

source='data/'
for directory in os.listdir(source):
    for file_ in os.listdir(source+directory):
        print(file_)
        dataset=pd.read_csv(f'{source+directory}/{file_}',names=['f1','f2','f3','f4','f5','ax','ay','az','gx','gy','gz'],header=None)
        df=pd.DataFrame(dataset)
        df[df<0]=0.00
        #print(df)
        for i in range(6):
            df.to_csv(str(i)+'file.csv',index=False,header=False)
        