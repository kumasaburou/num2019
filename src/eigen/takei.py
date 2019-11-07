## べき乗法
固有値の最大値に対する固有ベクトルを求める

   $$\boldsymbol{A}\boldsymbol{x}=\lambda\boldsymbol{x}$$

$$
\boldsymbol{x^{(1)}} = \boldsymbol{A}\boldsymbol{x^{(0)}}\\
\boldsymbol{x^{(2)}} = \boldsymbol{A}\boldsymbol{x^{(1)}}\\
\vdots\\
\boldsymbol{x^{(k+1)}} = \boldsymbol{A}\boldsymbol{x^{(k)}}\\ = \boldsymbol{A^k}\boldsymbol{x^{(0)}}\\
$$

$\boldsymbol{x^{(k)}}$は絶対値が最大の固有値に対応する固有ベクトルに収束する。（発散しないようにベクトルの大きさは一定）

任意のベクトル$\boldsymbol{y^{(0)}}$
$$
\boldsymbol{y^{(0)}} = c_1\boldsymbol{x_1} + c_2\boldsymbol{x_2} +\dots +c_n\boldsymbol{x_n}\\
(|\lambda_1| > |\lambda_2| > |\lambda_3| > \dots \geqq |\lambda_n|)\\
\quad\\
\boldsymbol{x^{(1)}} = \boldsymbol{A}\boldsymbol{y^{(0)}} =\lambda_1c_1\boldsymbol{x_1}+\lambda_2c_2\boldsymbol{x_2}+\dots+\lambda_nc_n\boldsymbol{x_n}\\
\boldsymbol{x^{(2)}} = \boldsymbol{A}\boldsymbol{x^{(1)}} =\lambda_1^2c_1\boldsymbol{x_1}+\lambda_2^2c_2\boldsymbol{x_2}+\dots+\lambda_n^2c_n\boldsymbol{x_n}\\
\vdots\\
\boldsymbol{x^{(k)}}=\boldsymbol{A^{k}}\boldsymbol{y^{(0)}} =\lambda_1^kc_1\boldsymbol{x_1}+\lambda_2^kc_2\boldsymbol{x_2}+\dots+\lambda_n^kc_n\boldsymbol{x_n}
$$

　$c_1\neq0$　なら

$$
\boldsymbol{y^{(k)}}=\lambda_1^k c_1 \{\boldsymbol{x}_1 + (\frac{\lambda_2}{\lambda_1})^k\frac{c_2}{c_1}\boldsymbol{x_2} + \dots + \{\frac{\lambda_n}{\lambda_1})^k\frac{c_n}{c_1}\boldsymbol{x_n}\}
$$

$\frac{\lambda_k}{\lambda_1}<0$なので$k\to\infty$で大きさが最大の固有値に対する固有ベクトルが得られる。

$$
\boldsymbol{y^{(k-1)}} = \lambda_1^{k-1}c_1\boldsymbol{x_1}\to\boldsymbol{x^{(k)}}=\frac{\boldsymbol{y^{(k-1)}}}{|\boldsymbol{y^{(k-1)}}|} = \lambda_1^{k-1}c_1\boldsymbol{x_1}\frac{1}{\boldsymbol{y^{(k-1)}}}\\
\boldsymbol{y^{(k)}} = \boldsymbol{A}\boldsymbol{x^{(k-1)}} =  \lambda_1^{k}c_1\boldsymbol{x_1}\frac{1}{\boldsymbol{y^{(k-1)}}}\\
\quad\\
\quad\\
\frac{(\boldsymbol{x^{(k)}},\boldsymbol{y^{(k)}})}{(\boldsymbol{x^{(k)}},\boldsymbol{x^{(k)}})} = \frac{((\lambda_1^{k-1}c_1\boldsymbol{x_1}\frac{1}{\boldsymbol{y^{(k-1)}}}),(\lambda_1^{k}c_1\boldsymbol{x_1}\frac{1}{\boldsymbol{y^{(k-1)}}}))}{((\lambda_1^{k-1}c_1\boldsymbol{x_1}\frac{1}{\boldsymbol{y^{(k-1)}}}),(\lambda_1^{k-1}c_1\boldsymbol{x_1}\frac{1}{\boldsymbol{y^{(k-1)}}}))} = \lambda_1
$$

                                       

import numpy as np

A = np.array([[1,2],[3,4]])
x0 = np.array([1,0]); x1 = np.array([0,1])
u = 1.0 * x0 + 2.0 * x1

rel_eig = 0.000001
rel_delta_u =10.0


while rel_delta_u >= rel_eig :
    uu = u/np.linalg.norm(u)
    print("u=",uu)
    
    u = np.dot(A,uu.T)
    
    eigen_value = np.dot(uu,u)/(np.dot(uu,uu.T))
    print("eig_val=",eigen_value)
    
    delta_u_vec = uu - u/np.linalg.norm(u)
    abs_delta_u_value = np.linalg.norm(delta_u_vec)
    rel_delta_u = abs_delta_u_value/np.abs(eigen_value)
    print("rel_delta_u = ",rel_delta_u)
                                    

## 逆べき乗法
$$
\boldsymbol{A}\boldsymbol{x}=\lambda\boldsymbol{x}\\
\to\boldsymbol{A^{-1}}\boldsymbol{A}\boldsymbol{x} = \lambda\boldsymbol{A^{-1}}\boldsymbol{x}\\
\to\boldsymbol{A^{-1}}\boldsymbol{x}=\lambda^{-1}\boldsymbol{x}
$$

$\boldsymbol{A'} = \boldsymbol{A^{-1}},\lambda'=\lambda^{-1}$
として同じように計算することで大きさが最小の固有値に対する固有ベクトルが得られる。

## 収束の加速
$$
\boldsymbol{A}\boldsymbol{x}=\lambda\boldsymbol{x},\boldsymbol{B}=\boldsymbol{A}-p\boldsymbol{I}\\
\boldsymbol{A}\boldsymbol{x}=(\boldsymbol{B} + p\boldsymbol{I})\boldsymbol{x} = \lambda\boldsymbol{x}\\
\boldsymbol{B}\boldsymbol{x}=\lambda\boldsymbol{x} - p\boldsymbol{I}\boldsymbol{x}=(\lambda - p)\boldsymbol{x}\\
$$
適当な$p$を選択することで固有値同士の比を大きくすることで収束を早くする。

$$
\frac{|\lambda_2-p|}{|\lambda_1-p|}<\frac{|\lambda_2|}{|\lambda_1|}
$$
             
             
import numpy as np

A = np.array([[1,2],[3,4]])
B = A - np.array([[1,0],[0,1]])
print(B)
x0 = np.array([1,0]); x1 = np.array([0,1])
u = 1.0 * x0 + 2.0 * 1

rel_eig = 0.000001


rel_delta_u =100.0
while rel_delta_u >= rel_eig :
    uu = u/np.linalg.norm(u)
    print("u=",uu)
    
    u = np.dot(A,uu.T)
    
    eigen_value = np.dot(uu,u)/(np.dot(uu,uu.T))
    print("eig_val=",eigen_value)
    
    delta_u_vec = uu - u/np.linalg.norm(u)
    abs_delta_u_value = np.linalg.norm(delta_u_vec)
    rel_delta_u = abs_delta_u_value/np.abs(eigen_value)
    print("rel_delta_u_vec = ",rel_delta_u)