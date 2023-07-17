import scipy.io as scio
data=scio.loadmat('./raw_data/drug_data/Gdataset/Gdataset.mat')
# print(data.keys())
print(data['Wrname'][0])

