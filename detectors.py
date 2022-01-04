import numpy as np
import cv2

debug = 0


class Detectors(object):  # 检测每一帧图像中的物体
    # Python中三引号"""可以：
    # 1.用来做多行注释（一般会被运行，不推荐）
    # 2.允许一个字符串跨行，字符串中可以包含换行符/制表符和其它特殊字符
    # 3.用于SQL语句（对数据库进行查询修改）表达式没有变量，为避免使用转义换行符\n时
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        """
        return

    def Detect(self, frame, frame_count):  # 输入：帧图像  输出：目标质心的向量
        # 检测帧图像中的目标可以分为以下几步：
        # 1.将抓取到的帧图像由BGR格式转换成灰度图像
        # 2.背景相减
        # 3.Canny算子检测边缘
        # 4.仅保留阈值范围内的边缘
        # 5.找到轮廓及质心
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # # Convert BGR to GRAY
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # #
        # count = frame_count
        # if count == 100:
        #     cv2.imwrite('5_gray1.bmp', gray)
        # if debug == 0:
        #    cv2.imshow('gray', gray)
        #
        # # Perform Background Subtraction 背景相减，得到前景掩膜
        # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma)
        # 使用K(K=3或5)个高斯混合分布对背景像素进行建模，使用这些颜色在视频中存在时间的长短作为混合的权重。每一个像素点基于时间序列会有很多值，从而可以构成一个分布
        # history：时间长度，默认200   nmixtures：高斯混合成份的数量，默认5
        # backgroundRatio：背景比率，默认0.7   noiseSigma：噪声强度（亮度或每个颜色通道的标准偏差），默认0表示一些自动值
        # fgbg = cv2.createBackgroundSubtractorMOG2()
        # fgmask = fgbg.apply(gray)

        # if debug == 0:
           # cv2.imshow('bgsub', fgmask)

        # Detect edges  Canny边缘检测算子
        # edges = cv.Canny(image, threshold1, threshold2[, apertureSize[, L2gradient]])
        # threshold1/2 处理过程中的两个阈值   apertureSize：Sobel算子的孔径大小
        # L2gradient：计算图像梯度幅度的标识，默认False，使用L1范数计算（将两个方向导数的绝对值相加），若为True，使用L2范数进行计算（两个方向的导数的平方和再开方）
        # edges = cv2.Canny(gray, 100, 190, 3)
        # #
        # if debug == 0:
        #    cv2.imshow('Edges', edges)


        # Retain only edges within the threshold
        # ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)


        # Step1：将BGR图像转换成Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # count = frame_count
        # if count==100:
        #     cv2.imwrite('100_gray.bmp', gray)

        # Step2：图像阈值分割
        # cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
        # src：输入图片 thresh：阈值  maxval：填充色 type：有01234四种类型，一般为0
        # 输出：retval：返回输入的thresh  dst：返回二值化后的图片
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Find contours
        # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(image,mode,method)查找物体的轮廓
        # mode：轮廓的模式  cv2.RETR_EXTERNAL只检测外轮廓 cv2.RETR_LIST检测的轮廓不建立等级关系
        # cv2.RETR_CCOMP建立两个等级的轮廓，上一层为外边界，内层为内孔的边界。如果内孔内还有连通物体，则这个物体的边界也在顶层
        # cv2.RETR_TREE建立一个等级树结构的轮廓。
        # method：轮廓的近似方法 cv2.CHAIN_APPROX_NOME存储所有的轮廓点，相邻的两个点的像素位置差不超过1；
        # cv2.CHAIN_APPROX_SIMPLE压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需要4个点来保存轮廓信息
        # cv2.CHAIN_APPROX_TC89_L1：使用teh-Chinl chain 近似算法保存轮廓信息。
        # image:跟输入图像类似的一张二值图
        # contours：list结构，list中每个元素用ndarry表示，是图像中一个轮廓的点的集合，是(x,1,2)的三维向量，x是该条边沿里有多少个像素点，2表示每个点的横纵坐标
        # hierarchy：(x,4)的二维ndarry。元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




        if debug == 0:
            cv2.imshow('thresh', thresh)

        centers = []  # vector of object centroids in a frame
        blob_radius_thresh = 10
        # Find centroid for each valid contours 找到目标轮廓边界后，寻找能覆盖各目标最小圆的中心及半径
        for cnt in contours:
            try:
                # Calculate and draw circle
                # (x, y), radius = cv2.minEnclosingCircle(cnt)使用迭代算法查找包含2D点集的最小区域的圆
                # (x,y):目标圆心  radius：半径
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if radius > blob_radius_thresh:  # 如果半径大于10，绘制覆盖该目标的圆
                    # cv2.circle(image, center_coordinates, radius, color, thickness)
                    # color:绘制的圆的边界的颜色，（255，0，0)是蓝色
                    # thickness:画圆的线条的粗细
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    # np.array:创建一个数组
                    b = np.array([[x], [y]])
                    # np.round_(arr, decimals = 0, out = None)将数组四舍五入为给定的小数位数
                    # arr：输入数组 decimal：[int]要舍入的小数位，默认值为0  out:[可选]输出结果数组
                    # np.round(data, decimal) 对于数据data，保留小数点后decimal位小数，默认值为0
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        # show contours of tracking objects
        # cv2.imshow('Track Bugs', frame)

        return centers
