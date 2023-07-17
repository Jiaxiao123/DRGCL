def get_mean_var_std(arr):
    import numpy as np

    # 求均值
    arr_mean = np.mean(arr)
    # 求方差
    arr_var = np.var(arr)
    # 求标准差
    arr_std = np.std(arr, ddof=1)
    print("平均值为：%f" % arr_mean)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)

    return arr_mean, arr_var, arr_std


if __name__ == '__main__':
    arr = [0.5588, 0.5553, 0.5795, 0.5691, 0.5648, 0.5718, 0.5547, 0.5551, 0.5663, 0.5733]
    print(get_mean_var_std(arr))
