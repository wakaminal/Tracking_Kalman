import numpy as np
# 在存在噪声信息干扰的情况下，在含有不确定信息的动态系统中，卡尔曼滤波能对系统下一步要做什么做出有根据的预测
# 卡尔曼滤波适合不断变化的系统，内存占用小（只需要保留前一个状态），速度快
class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class"""
        self.dt = 0.005 # delta time  # 采样时间

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations 观测矩阵
        self.u = np.zeros((2, 1))  # previous state vector x

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations y
        # np.diag(v,k=0)以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换成方阵（非对角线元素为0）
        # v:array_like 如果v是二维数组，返回k位置的对角线；如果v是一维数组，返回一个v作为k位置对角线的二维数组
        # k:int,optional  可选参数，默认为0.代表对角线的位置，大于0位于对角线上面，小于0则在下面
        # 如：x=array([[0,1,2],[3,4,5],[6,7,8]])->np.diag(x)=array([0,4,8]) np.diag(x,k=1)=array([1,5])
        # np.diag(x,k=-1)=([3,7])  np.diag(np.diag(x))=([[0,0,0],[0,4,0],[0,0,8]])
        self.P = np.diag((3.0, 3.0))  # covariance matrix 协方差矩阵
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat 状态转移矩阵
        # numpy.eye(N,M=None,k=0,dtype=<class 'float'>,order='C) 返回二维数组（N,M)，对角线位置是1，其余地方为0
        # k：对角线的下标，0表示主对角线，负数表示低对角线，正数表示高对角   dtype：可选项，返回的数据的类型
        # self.u.shape[0]读取矩阵u第一维度的长度
        self.Q = np.eye(self.u.shape[0])  # process noise matrix 过程噪声协方差矩阵
        self.R = np.eye(self.b.shape[0])  # observation noise matrix 观测噪声协方差矩阵
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            # 预测状态向量x和协方差P
            u(k+1)=F*u(k)+
            b(k)=C(k)u(k)+w(k)
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose  # F.T是F的转置
        """
        # Predicted state estimate
        # np.dot->向量内积 多维矩阵乘法 矩阵与向量的乘法
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance

        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:    # update using prediction
            self.b = self.lastResult
        else:           # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R  # C = AP_{k|k-1} A.T + R
        # np.linalg.inv()矩阵求逆
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))  # K_{k} = P_{k|k-1} A.T(C.Inv)
        #  u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A, self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))  # P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
        self.lastResult = self.u
        return self.u
