import numpy as np
from kalman_filter import KalmanFilter
from common import dprint
from scipy.optimize import linear_sum_assignment
# sklearn里的linear_assignment()以及scipy里的linear_sum_assignment()都实现了匈牙利算法，两者的返回值的形式不同
# sklearn API result:           scipy API result:
# [[0 1]                        (array([0, 1, 2], dtype=int64), array([1, 0, 2], dtype=int64))
# [1 0]
# [2 2]]

class Track(object):
    """Track class for every object to be tracked"""

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked"""

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh 最大跟踪长度: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip 未检测到跟踪对象允许跳过的最大帧数: maximum allowed frames to be skipped for the track object undetected
            max_trace_length 跟踪路径历史长度: trace path history length
            trackIdCount 用于识别每个追踪对象: identification of each track object
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids  # 使用预测与检测到的质心的平方距离总和计算成本
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks  # 使用匈牙利算法将正确检测到的测量值分配给预测轨迹
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        """

        # Create tracks if no tracks vector found
        if len(self.tracks) == 0:  # 如果跟踪的目标数是0
            for i in range(len(detections)):  # 将检测的目标中心点代入，初始化追踪函数
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between predicted and detected centroids
        # 计算检测与跟踪目标相关联的代价矩阵，预测位置与探测到的目标之间的距离作为代价
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] + diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        # Let's average the squared ERROR
        cost = 0.5 * cost

        # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
        # 使用匈牙利算法将前一帧中的跟踪框tracks与当前帧中的检测框detections进行关联
        # 通过外观信息（appearance information）和马氏距离（Mahalanobis distance），或者IOU来计算代价矩阵。
        assignment = []
        for _ in range(N):  # 初始化跟踪个数
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)  # 分别得到最小代价的行/列
        for i in range(len(row_ind)):  # 将最小代价矩阵得到的行和列对应，也就是说可以关联的赋给assignment
            assignment[row_ind[i]] = col_ind[i]  # 此时assignment中等于-1的就是探测和预测到的目标数量不一致，无法关联

        # 探测目标与跟踪目标无法关联存在两种情况：
        # 1.一开始匈牙利算法就未关联到（探测目标个数！=追踪目标个数，使有的目标点被落下），
        # 2.关联到但彼此间距离超过所设置的阈值
        # 即追踪中出现新目标，或原目标消失
        '''或许还有一种可能，目标移动速度过快导致代价矩阵所关联的目标匹配混乱，或许会超过设置的追踪距离阈值，使判别为不关联'''
        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:  # 如果追踪的第i个目标与检测目标可关联
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1  # 如果前后能关联，但彼此之间距离超过所设置的追踪距离的上限，设置其为不可关联
                    un_assigned_tracks.append(i)  # 扩展追踪过程无关联的列表
                pass
            else:
                self.tracks[i].skipped_frames += 1  # 追踪的第i个目标未出现的帧数+1

        # If tracks are not detected for long time, remove them
        # 长时间未跟踪到的目标，从跟踪和关联中移去
        del_tracks = []  # 建立一个取消跟踪的数组
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)  # 如果连续多帧未跟踪到正在跟踪的目标i，就将其添加到不再跟踪的数组
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:  # 如果存在不需要再跟踪的目标，将其从跟踪数组tracks[]中删去
                if id < len(self.tracks):
                   del self.tracks[id]
                   del assignment[id]
                else:
                   dprint("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        # 未关联到的探测目标
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks for unassigned detects
        # 未关联探测目标的重新进行跟踪
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                # 使用观测值
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]], 1)
            else:
                # 使用预测值
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)
            length = len(self.tracks[i].trace)
            if length > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)  # 拓展跟踪路径
            self.tracks[i].KF.lastResult = self.tracks[i].prediction  # 更新卡尔曼滤波状态
