# 导入Excel数据studentshuju.xls的代码：
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from matplotlib.pyplot import MultipleLocator
import numpy as np
from matplotlib import rcParams
import xlrd
import seaborn as sns

path = "../Contrastive_learn/PLOT/Gdataset_disease_denovo.csv"
df = pd.read_csv(path)
print(df)

###做箱式图的代码：
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.boxplot(df['topology'])
# ax.boxplot((df['topology'], df['feature']),labels=('topology','feature'))
# plt.xlabel('space')
# plt.ylabel('Attention value')
# plt.title('Gdataset_dis')
# # plt.show()
# plt.savefig('')

###做折线图

mpl.rcParams['font.family'] = ['Times New Roman']   # 添加这条可以让图形显示中文

# plt.plot(df["beta"], df["G_AUC"], label='Gdataset', linewidth=1, color='#F27970', marker='o', markerfacecolor='#F27970',
#          markersize=5)
# plt.plot(df["beta"], df["C_AUC"], label='Cdataset', linewidth=1, color='#54B345', marker='*', markerfacecolor='#54B345',
#          markersize=5)
# plt.plot(df["beta"], df["L_AUC"], label='Ldataset', linewidth=1, color='#05B9E2', marker='D', markerfacecolor='#05B9E2',
#          markersize=5)
# plt.plot(df["beta"], df["lrssl_AUC"], label='LRSSL', linewidth=1, color='#8983BF', marker='^', markerfacecolor='#8983BF',
#          markersize=5)
# x = [1,2,3,4,5,6]
# plt.plot(x, df["G_AUPR"], label='Gdataset', linewidth=1, color='#F27970', marker='o', markerfacecolor='#F27970',
#          markersize=5)
# plt.plot(x, df["C_AUPR"], label='Cdataset', linewidth=1, color='#54B345', marker='*', markerfacecolor='#54B345',
#          markersize=5)
# plt.plot(x, df["L_AUPR"], label='Ldataset', linewidth=1, color='#05B9E2', marker='D', markerfacecolor='#05B9E2',
#          markersize=5)
# plt.plot(x, df["lrssl_AUPR"], label='LRSSL', linewidth=1, color='#8983BF', marker='^', markerfacecolor='#8983BF',
#          markersize=5)
# for a, b in zip(df["β"], df["auroc_G"]):  # 添加这个循环显示坐标
#     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=)
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签


# group_labels = ['0.001', '0.01','0.1','1','10','100']
# plt.xticks(x, group_labels, rotation=0)
# plt.xticks([2,4,6,8,10,12,14,16,18,20])  #指定x轴刻度的数目与取值
# plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])


# # rcParams.update(config)
# plt.xlabel('β:the contrastive loss coefficient')
# # plt.xlabel('k:the number of neighbors')
# # plt.xlabel('τ:the temperature parameter')
# plt.ylabel('AUPRC')
# # plt.title("β-parameter")
# plt.legend(loc="best")
# plt.grid()
#
# # plt.show()
# plt.savefig('AUPRC_β.pdf')  # 保存该图片

### 做散点图的代码：
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# x = df['xuhao']
# y = df['height']
# z = df['weight']
# ax.scatter(x, y, c='b', marker='s', s=30)
# ax.scatter(x, z, c='y', marker='o', s=60)
# plt.show()

###做饼图的代码：
# var = df.groupby(['sex']).sum().stack()
# t = var.unstack()
# type(t)
# x = t['age']
# label1 = t.index
# plt.axis('equal')
# plt.pie(x, labels=label1, autopct='%1.1f%%')
# plt.show()
# mpl.rcParams['font.family'] = ['Times New Roman']  # 添加这条可以让图形显示中文
#柱形图


bar_width = 0.5
color = ['#A1A9D0','#F0988C','#B883D4','#9E9E9E','#CFEAF1', '#C4A5DE', '#F6CAE5']
plt.bar(df["model"], df["auroc_mean"], width=bar_width, color=color)

tick_label=df["model"]
# plt.bar(x + bar_width, df["aupr"], tick_label=df["Variants"], width=bar_width)
# for a,b in zip(df["model"], df["auroc_mean"]):   #柱子上的数字显示
#  plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=15);
# plt.legend(loc="best")

plt.xticks(rotation=30)
plt.ylabel("AUROC")
plt.ylim(0.50,0.95)
# plt.ylim(0.65,0.95)
# plt.ylim(0.00,0.40)
plt.savefig('auroc_disease_denovo.pdf')
plt.show()



