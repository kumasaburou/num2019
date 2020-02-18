import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def initial_config(size_lat):
    "初期スピン配位の生成"
    return 2*(np.random.randint(0, 2, size=(size_lat, size_lat))-1/2)

def hamiltonian(arr_spin, coupling, ex_field):
    "Isingモデルのハミルトニアン（交換相互作用と外場）"
    n0, n1 = np.shape(arr_spin)
    energy = 0.0
    for i in range(-1,n0-1):
        for j in range(-1,n1-1):
            energy += -coupling*(arr_spin[i][j]*arr_spin[i][j+1] + arr_spin[i][j]*arr_spin[i+1][j])
    energy += -ex_field*np.sum(arr_spin)
    return energy

def update_config(arr_spin, n0, n1, energy, temp, coupling, ex_field):
    """
    メトロポリス法にしたがってスピン配位を更新するサブルーチン
    副作用：フリップによるエネルギーの変化、スピン配位の更新
    """
    for i in range(0,n0):
        for j in range(0,n1):
            diff_energy = diff_hamiltonian(arr_spin, n0, n1, i, j, coupling, ex_field)
            # diff_energy<=0であれば指数函数の値は1以上なので必ず採択
            # diff_energy>0であれば乱数の値によって採択されるかされないかが決まる
            # 正のdiff_energyが小さいほど採択率が高い
#             # 短いけどちょっと時間がかかる
#             if np.random.rand() <= np.exp(-diff_energy/temp):
#                 spin_flip(arr_spin, i, j)
#                 energy += diff_energy
            # 上のifを分解したif-ifel
            if diff_energy <= 0:
                energy += diff_energy
                spin_flip(arr_spin, i, j)
            elif np.random.rand() < np.exp(-diff_energy/temp):
                energy += diff_energy
                spin_flip(arr_spin, i, j)


def diff_hamiltonian(arr_spin, n0, n1, i, j, coupling, ex_field):
    "指定のスピンをフリップしたときに生じるハミルトニアンの差分をスピン配位を変更せずに出力"
    return 2*(coupling*(arr_spin[i][j]*arr_spin[i-1][j] + arr_spin[i][j]*arr_spin[-n0+i+1][j] 
                        + arr_spin[i][j]*arr_spin[i][j-1] + arr_spin[i][j]*arr_spin[i][-n1+j+1])
              + ex_field*arr_spin[i][j])

def spin_flip(arr_spin, i, j):
    "指定したスピンをフリップするサブルーチン"
    arr_spin[i][j] = -arr_spin[i][j]

def magnetization_plot(size_lat, temp_min, temp_max, coupling, ex_field):
    """
    温度に対する1サイト当たりの磁化の計算とその温度-磁化グラフのプロット
    T_c = 2J/log(sqrt(2)+1) ~ 2.27J
    """
    num_update = 1000 # 最大更新回数
    num_update_eq = 200 # 最大更新回数のうち熱平衡へ移すための更新回数
    mag_lst =[] # 各温度における1サイトあたりの磁化のリスト
    mag_samples = [] # 各温度における1サイトあたりの磁化のサンプルのリスト
    temp_step = 100 # プロットする温度の分割数
    temp_lst = np.flip(np.linspace(temp_min, temp_max, temp_step)) # プロットする温度のリスト
    
    arr_spin = initial_config(size_lat) # 初期スピン配位生成
    n0, n1 = np.shape(arr_spin)
    energy = hamiltonian(arr_spin, coupling, ex_field) # 初期スピン配位におけるエネルギー

    for temp_tmp in temp_lst:
        mag_samples = []
        # 熱平衡までの更新
        for i in range(0, num_update_eq):
            update_config(arr_spin, n0, n1, energy, temp_tmp, coupling, ex_field)
        # 熱平衡状態での平均磁化のサンプル取得
        for i in range(0, num_update - num_update_eq):
            update_config(arr_spin, n0, n1, energy, temp_tmp, coupling, ex_field)
            mag_samples.append(np.sum(arr_spin)/len(arr_spin)**2)

        mag_lst.append(np.sum(mag_samples)/len(mag_samples))
    
    # プロット
    plt.plot(temp_lst, mag_lst)
    plt.grid(True)
    plt.ylabel("magnetization per cite")
    plt.xlabel("T")
    plt.xlim(round(temp_min,1), temp_max)
    plt.show()