import pandas as pd
import matplotlib.pyplot as plt



dataset=pd.read_csv('tinyml/file.csv',names=['f1','f2','f3','f4','f5','ax','ay','az','gx','gy','gz'],header=None)

df=pd.DataFrame(dataset)
print(df)
df[df<=0]=0
print(df)
print(plt.style.available)

plt.style.use('bmh')
plt.plot(df.index,df['f1'])
plt.plot(df.index,df['f2'])
plt.plot(df.index,df['f3'])
plt.plot(df.index,df['f4'])
plt.plot(df.index,df['f5'])
plt.plot(df.index,df['ax'],color='y',marker='o')
plt.plot(df.index,df['ay'])
plt.plot(df.index,df['az'])
plt.plot(df.index,df['gx'])
plt.plot(df.index,df['gy'])
plt.plot(df.index,df['gz'],color='k',linestyle='--' ,marker='.')

plt.legend(['f1','f2','f3','f4','f5','ax','ay','az','gx','gy','gz'])

plt.show()
print(df['ax'].describe())

def count_letter(text):
    result={}
    for letter in text:
        if letter not in text:
            result[letter]=0
     
        result[letter]+=1
    return result

count_letter('hello there i am using whatsapp')