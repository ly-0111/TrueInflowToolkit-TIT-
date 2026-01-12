import numpy as np
import pandas as pd
from D1 import HD,recreate_txt_file,append_data_to_txt,run
import time,copy
import multiprocessing
from D1runlevel import D1run,reset_file,ff_d,add_column_to_csv,head_thred,flow_thred
import csv
import datetime
import psutil
import os
import shutil
import matplotlib.pyplot as plt
class KalmanFilter:
    def __init__(self, obs, host, initZ,lateralflow,lentime):
        """
        初始化卡尔曼滤波器

        参数:
        F: 状态转移矩阵
        H: 观测矩阵
        Q: 状态转移协方差矩阵
        R: 观测噪声协方差矩阵
        x0: 初始状态估计
        P0: 初始状态协方差矩阵
        """
        
        #======================== parameter:   boundary of D1 flow and level
        # self.flow = copy.copy(bound)
        # self.wL = bound[1]
        self.simtime = 6 #h #在主循环中改变
        self.waitingtime = 2
        self.num_cores = 10
        self.listnum = []
        #======================== observation: level
        self.num = 1#后面不要几个水位
        self.numb = 0#前面不要几个水位
        # self.obs = copy.copy(obs[1+self.numb:-self.num-1])
        self.obs = copy.copy(obs[:])
        #======================== obs gauss: obs
        self.mean1 = 0  # 均值
        self.std_dev1 = 0.01  # 标准差
        self.std_dev1lst = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.num_samples = len(self.obs)  # 样本数
        
        #========================
        #======================== sim gauss: sim
        self.mean2 = 0  # 均值
        self.std_dev2 = self.std_dev1  # 标准差
        self.num_samples = len(self.obs)  # 样本数
        #========================
        # self.ZS = [obs[-1] for x in bound] #末水位 
        self.time = 0.25 #h
        #========================
        self.Q_ = copy.copy(host[0])
        #初始水位
        self.Zend = copy.copy(obs[-2])#下游边界
        #========================
        self.initZ = initZ
        self.lateralflow = lateralflow#支流流量
        self.lentime = lentime
        self.maxflow = 0
        self.outflow = []
        self.num_cores = 0
        self.hottime = 0
        self.hotoption1 = 1
        self.inflow =[]
        self.qt = []
        self.xx = []
        self.indexhead = []
        self.obsitmulti = []
        self.nha = 0
        self.ha = []
    def gauss1(self,std_dev,n):
        
        mean = copy.copy(self.mean1)
        # std_dev = self.std_dev1
        # num_samples = self.num_samples
        # np.random.seed(0)
        self.noise1 = []
        # for i in range(num_samples):
        self.noise1 = np.random.normal(mean, std_dev, n)
        
    def gauss2(self):
        
        mean = self.mean2
        std_dev = self.std_dev2
        num_samples = self.num_samples
        # np.random.seed(0)
        self.noise2 = np.random.normal(mean, std_dev, num_samples)  
    
    def D1_prediction(self,q,Man,u,n):
        
        """
        一维水动力：     
        Qflow:          输出各断面流量
        water_level:    输出各断面水位
        """
        
        upper_threshold = max(initZ+100) #水位错误上限
        lower_threshold = min(initZ-100)#水位错误下限
        time = self.time #同化时间间隔
        self.x = []
        self.p = []
        self.Manning = []
        i = 0
        num_cores = multiprocessing.cpu_count()
        parallel_count = num_cores = self.num_cores
        pool = multiprocessing.Pool(processes=num_cores)
        qt = [[] for i in range(len(self.qt))]
        boundqt = []
        print('meanq',q[0])
        while len(self.p) < n:
            # i = i+1
            nm = n - len(self.p)
            if i > 10 :
                i = 10
            q = [0 if x < 0 else x for x in q]
            boundlaterflow = []
            # print(q,u)
            if nm < 0:
                nm =0
            for ii in range(len(self.qt)):
                # qt[ii] = np.random.normal(self.qt[ii], u[0]*(1+i/5+ii/3), nm)   
                qt[ii] = np.random.normal(int(q[0]), u[0]*(1+i/5), nm)
            bound33 = np.random.normal(q[1], u[1]*(1+i/5), nm)
            bound44 = np.random.normal(q[2], u[2]*(1+i/5), nm)
            bound55 = np.random.normal(q[3], u[3]*(1+i/5), nm)
            bound66 = np.random.normal(q[4], u[4]*(1+i/5), nm)
            bound77 = np.random.normal(q[5], u[5]*(1+i/5), nm)
            bound88 = np.random.normal(q[6], u[6]*(1+i/5), nm)
            bound99 = np.random.normal(q[7], u[7]*(1+i/5), nm)
            bound100 = np.random.normal(q[8], u[8]*(1+i/5), nm)
            boundlaterflow = [bound33,bound44,bound55,bound66,bound77,bound88,bound99,bound100]
#---------------------------------------------------------------transfor numpy to list and add some schemes for judge Head arrival time
                           
            maxIndexl = []
            minIndexl = []
            #------------------------------ main stream
            bound = [x for x in qt[0]] #数量也是方案的数量
            for ii in range(len(self.qt)):
                qt[ii] = [x for x in qt[ii]]
            for ii in range(len(self.qt)):
                if ii == 0:
                   qt[ii].append(max(qt[ii])+u[0])
                   bound.append(max(qt[ii])+u[0])
                   maxIndexl.append(len(qt[ii])-1)
                   qt[ii].append(min(qt[ii])-u[0])
                   bound.append(min(qt[ii])-u[0])
                   minIndexl.append(len(qt[ii])-1)
                else:
                    qt[ii].append(int(q[0]))
                    qt[ii].append(int(q[0]))
            
            #------------------------------ tributary
            for ii in range(len(boundlaterflow)):
                if ii in self.listnum:
                    boundlaterflow[ii] = [x for x in boundlaterflow[ii]]
                else:
                    boundlaterflow[ii] = [0 for x in boundlaterflow[ii]]
            for ii in range(len(boundlaterflow)):
                boundlaterflow[ii].append(q[ii])
                boundlaterflow[ii].append(q[ii])
            for ii in range(len(boundlaterflow)):
                if ii in self.listnum:
                    for i2 in range(len(self.qt)):
                        qt[i2].append(qt[i2][1])
                        qt[i2].append(qt[i2][1])
                    for i3 in range(len(boundlaterflow)):
                        if i3 == ii:
                            boundlaterflow[i3].append(max(boundlaterflow[i3])+u[i3])
                            maxIndexl.append(len(boundlaterflow[i3])-1)
                            boundlaterflow[i3].append(min(boundlaterflow[i3])-1)
                            minIndexl.append(len(boundlaterflow[i3])-1)
                        else:
                            boundlaterflow[i3].append(0)
                            boundlaterflow[i3].append(0)
                            
            boundqt = [qt[kk][jj] for jj in range(len(qt[0])) for kk in range(len(self.qt))]
            #------------------------------ tributary
#---------------------------------------------------------------transfor numpy to list and add some schemes for judge Head arrival time            
            # if self.hotoption1 == 1:
            #     if maxIndex < self.num_cores :
            #         maxIndex=maxIndex + self.num_cores
            #         if maxIndex  == minIndex:
            #             maxIndex  = maxIndex + 1
            #         qt[0][maxIndex] = maxValue
            #     if minIndex < self.num_cores :
            #         minIndex=minIndex + self.num_cores
            #         if minIndex  == maxIndex:
            #             minIndex  = minIndex + 1
            #         qt[0][minIndex] = minValue

            lateralbound = boundlaterflow
            lateraltoD1 = []
            
            for k in range(len(qt[0])):
                lateral = []
                
                for j in range(len(self.lateralflow)):
                    lateral.append(self.lateralflow[j] + [lateralbound[j][k]]+[self.lateralflow[j][-1] for g in range(self.simtime*4)]+[lateralbound[j][k]])
                lateraltoD1.append(lateral)
 
            i = i+1
            self.flow = boundqt
            self.ZS = [self.Zend for x in qt[0]]
            j = 0
            tasksQ = self.flow#样本流量
            tasksZ = self.ZS#初始水位-上时刻观测-目前的固定值
            sub_Q_lists = [tasksQ[i:i+parallel_count*len(self.qt)] for i in range(0, len(tasksQ), parallel_count*len(self.qt))]
            sub_Z_lists = [tasksZ[i:i+parallel_count] for i in range(0, len(tasksZ), parallel_count)]
            sub_laterQ_lists = [[lateraltoD1[j] for j in range(i,min(i+parallel_count,len(lateraltoD1)))] for i in range(0, len(qt[0]), parallel_count)]
            self.Z_ = copy.copy(host[1]) + [obs[-2] for i in range(self.simtime*4)]


            self.outflow = host[2] + [self.maxflow]
            hotoption = 1
            resulthead= [] #所有水位
            resultoutf = []
            nqt = len(self.qt)
            for sub_Q_list, sub_Z_list,sub_laterQ_list in zip(sub_Q_lists, sub_Z_lists,sub_laterQ_lists):
                results = pool.starmap(D1run, [(j, num_cores,hotoption,self.hotoption1,self.hottime,sub_Q_list[j*nqt:(j+1)*nqt], self.Q_+sub_Q_list[j*nqt:(j+1)*nqt]+[self.Q_[-1] for i in range(self.simtime*4)], self.Z_, self.initZ, sub_laterQ_list[j], time,self.lentime,self.waitingtime,self.listnum,self.outflow,backdt) for j in range(len(sub_laterQ_list))])
                results_dict = dict(results)
                hotoption = 0
                
                for j in range(len(results)):
                    self.p.append(results_dict[str(j)][0])
                    resulthead.append(results_dict[str(j)][4])
                    resultoutf.append(results_dict[str(j)][3])
            # print(resultoutf[11][0])
            # print(resulthead[0][0])
            # for re in range(len(resulthead[0])):
                # add_column_to_csv('sim/dataouthead.csv','sim/dataouthead.csv',str(re),resulthead[0][re].tolist())
            # exit()
            # add_column_to_csv('sim/dataoutflow.csv','sim/dataoutflow.csv',str(loop),resultoutf[0][0].tolist())
            # #-------------------
            # exit()
            # self.x = np.array(self.x)
            # print(self.p)
            # print(maxIndexl,minIndexl)
            # exit()
            indexhead = []
            indexheadml = []
            indexml = []
            # print(len(resulthead))
            # print(maxIndexl)
            # print(minIndexl)
            if not indexhead:
#------------------- deciside the time of flow 
            # print('res',resulthead[maxIndex][3],resulthead[minIndex][3])
                for ii in range(len(maxIndexl)):
                    indexhead = []
                    for id in range(len(resulthead[0])):
                        index = ff_d(resulthead[maxIndexl[ii]][id],resulthead[minIndexl[ii]][id],dq)[0]
                        dh = ff_d(resulthead[maxIndexl[ii]][id],resulthead[minIndexl[ii]][id],dq)[1]
                        if index == None:
                            index = len(resulthead[maxIndexl[ii]][id])-self.nha-1
                        if index > len(resulthead[maxIndexl[ii]][id])-self.nha-1:
                            index = len(resulthead[maxIndexl[ii]][id])-self.nha-1
                        indexhead.append(index)
                    indexheadml.append(indexhead)
                print(indexheadml)
                # print(indexheadml)
                # print(indexheadml[:][1])
                # exit()
                for i4 in range(len(resulthead[0])):
                    if i4 <= 0:
                        indexml.append(indexheadml[0][i4])
                    else:
                        indexml.append(min([lst[i4] for lst in indexheadml]))
                # for 
                indexheadmean.append(indexml)
                # print(indexheadmean)
                # indexhead = [12,21,27,30,33,37,40,43,47]
                if len(indexheadmean)<5:
                    indexhead = np.mean(indexheadmean,axis=0)
                if len(indexheadmean)>=5:
                    del indexheadmean[0]
                    indexhead = np.mean(indexheadmean[-5:],axis=0)
                
                indexhead=indexhead.tolist()
                indexhead = [int(x) for x in indexhead]
            if self.ha == []: 
                self.ha = [j1 for j1 in range(len(indexhead))]
                if self.num == 0:
                    self.ha = self.ha[self.numb:]
                else:
                    self.ha = self.ha[self.numb:-self.num]                    
                self.ha = [item for item in self.ha if item not in hdelete]
            if self.num == 0:
                indexhead = indexhead[self.numb:]
            else:
                indexhead = indexhead[self.numb:-self.num]
            
            #------------test for laterflow
            # indexhead[0] = 14
            # indexhead[1:] = [1]*(len(indexhead[1:]))
            #------------test for laterflow
            self.indexhead = indexhead
                
            
            # print(self.p[maxIndex][0],self.p[minIndex][0])
            print(indexhead)
            append_data_to_txt("sim/time.txt",[min(indexhead)])
            for i in range(len(resulthead)-len(maxIndexl)*2):#方案数
                headr = []
                for k in range(self.nha):#时间数
                    for ii in range(len(indexhead)):#水位断面数
                        if ii+self.numb not in hdelete:
                            headr.append(resulthead[i][ii+self.numb][indexhead[ii]+k])
                # headr = [resulthead[i][ii+self.numb][indexhead[ii]+k] for k in range(self.nha) for ii in range(len(indexhead)) ] #第i个结果的第ii个断面的第**时刻
                self.xx.append(headr)
            self.x = np.array(self.xx)
            
            # exit()
            # print(head260)
            # for index, row in enumerate(head260):
            #     plt.plot(row, label=f'行 {index+1}')    

            #     # 添加图的标题和标签
            #     plt.title('各行折线图')
            #     plt.xlabel('索引')
            #     plt.ylabel('值')
            #     plt.legend()
            #     plt.show()
            # print('self.p',self.p)
            # print('self.pmin',min(self.p))
            # print('self.pmax',max(self.p))
            #========================删除计算错误的行
            # rows_with_nan = np.any(np.isnan(self.x), axis=1)
            indices = np.where(np.any((self.x < lower_threshold) | (self.x > upper_threshold)|(np.isnan(self.x)), axis=1))
            # array_no_nan = self.x[~rows_with_nan]
            new_array = self.x[~np.any((self.x < lower_threshold) | (self.x > upper_threshold)|(np.isnan(self.x)), axis=1)]
            array_no_nan = new_array
            self.x = array_no_nan
            self.num_rows_with_nan = len(indices[0])
            print('Number of rows to be removed=',self.num_rows_with_nan)
            # indices_with_nan = np.where(rows_with_nan)[0]
            indices_with_nan = indices[0]
            # print('Indices of rows to be removed=',indices_with_nan)
            self.p = np.delete(self.p, indices_with_nan, axis=0)

            self.x = self.x.tolist()
            
            self.p = self.p.tolist()
            self.p = self.p[:-2*len(maxIndexl)]
            print(len(self.p))
        # self.p = np.array([self.p]).T #只有一个p时
        self.p = np.array(self.p) #有多个p时
        self.x = np.array(self.x)
        # print('self.x',self.x)
        self.obs = []
        for  k in range(self.nha):
            for i in range(len(self.indexhead)):
                if i+self.numb not in hdelete:
                    self.obs.append(self.obsitmulti[self.indexhead[i]+k+1][i+1+self.numb])#obsitmulti 第一列为流量，从第二列开始为水位
                    #self.obs.append(self.obsitmulti[self.indexhead[i]+k+1][i+1+self.numb])#obsitmulti 第一列为流量，从第二列开始为水位
        self.num_samples = len(self.obs)
        print(self.x[0],self.obs)
        # print(self.num_samples)
        return
    def calculate_average_difference(self):
        """
        计算两个列表对应位置差值的平均值。

        参数:
            list1 (list): 第一个列表。
            list2 (list): 第二个列表。

        返回:
            float: 差值的平均值。
        """
        # 检查两个列表的长度是否相同
        list1 = copy.copy(self.x[0])
        list2 = copy.copy(self.obs)
        if len(list1) != len(list2):
            raise ValueError("两个列表的长度必须相同")
        
        array1 = np.array(list1)
        array2 = np.array(list2)

        # 计算绝对差值并求平均值
        average_difference = np.mean(np.abs(array1 - array2))
        Galf = 1-0.001/average_difference
        if Galf < 0:
            Galf = 0
        return Galf
    def observation(self):
        
        """
        观测方程：       多个断面的实测水位
        gaussian_noise: 高斯噪声
        mean:           均值
        std_dev:        方差
        num_samples:    样本数=观测值数
        self.H:         观测矩阵 单位矩阵
        y:              观测输出
        """
        self.y = []
             
        for i in range(self.num_samples):
            self.gauss1(self.std_dev1,len(self.p))
            self.noise = self.noise1
            y = [ self.obs[i] + n for n in self.noise]
            
            self.y.append(np.array(y))
            
        self.y = np.array(self.y).T

        
    def covd(self,x,y):
        """
        每一列为一个变量
        n: 样本数量
        C: 协方差矩阵
        a: 
        """

        xmean = np.mean(x,axis=0)
        ymean = np.mean(y,axis=0)
        
        C = np.dot((x-xmean).T,y-ymean)/(x.shape[0] - 1)
        
        
        return C
        
    def K(self):
        
        """
        x:              预测矩阵
        y:              观测矩阵
        Sk:             观测误差协方差
        Cpy:            参数和观测的协方差
        Cyy:            观测和观测的协方差
        k:              卡尔曼增益
        np.cov:         求协方差
        np.linalg.inv:  求逆
        np.dot:         点成
        """
        
        y  = copy.copy(self.y)
        x  = copy.copy(self.x)
        p  = copy.copy(self.p)
 
        Cxy = np.array(self.covd(x, x))
        Cyy = np.array(self.covd(x, x))
        Cpy = np.array(self.covd(p, x))
        
        self.Cpy = Cpy
        self.Cyy = Cyy
        self.Cxy = Cxy
        
        YY = np.eye(self.y.shape[1]) * self.std_dev1*self.std_dev1

        kx  = np.dot(Cxy,np.linalg.pinv(Cyy+YY))
        kp  = np.dot(Cpy,np.linalg.pinv(Cyy+YY))
        
        self.kx = copy.copy(np.array(kx))
        self.kp = copy.copy(np.array(kp))
    def update(self):
        
        """
        x_: 状态先验
        p_: 参数先验
        _x: 状态后验
        _p: 参数后验
        """
        
        kp = np.copy(self.kp)
        kx = np.copy(self.kx)
        
        x_ = np.copy(self.x) 
        p_ = np.copy(self.p)
        
        # print(head_thred(self.y.T -  x_.T))
        # head_o = head_thred(self.y.T -  x_.T)
        # kp = flow_thred(kp,self.ha,listnum)
        print('kp',kp)
        # print(self.ha)
        # exit()
        _x = x_.T + np.dot(kx, (self.y.T -  x_.T))
        _p = p_.T + np.dot(kp, (self.y.T -  x_.T))
        
        # _x = x_.T + np.dot(kx, head_o)
        # _p = p_.T + np.dot(kp, head_o)
          
        _x = copy.copy(_x.T)
        _p = copy.copy(_p.T)

        self._Cpy = np.array(self.covd(_p, _x))
        # _p[_p < 0] = 0
        self.x_ = copy.copy(self.x)
        self.p_ = copy.copy(self.p)
        self._x = copy.copy(_x)
        self._p = copy.copy(_p)
   
        self.Cpp_ = self.covd(p_, p_)
        self._Cpp = self.covd(_p, _p)
        
        print('Cp_ = ',self.covd(p_, p_))
        print('C_p = ',self.covd(_p, _p))
        # print('_x',np.mean(x_,axis=0))
        # print('kp',kp)
        
        return np.mean(_x,axis=0),np.mean(_p,axis=0),np.mean(self.x_,axis=0)
    
def find_nan_indices(lst):
    array = np.array(lst)
    nan_indices = np.where(np.isnan(array))[0]
    return nan_indices.tolist()

def sum_of_absolute_values(numbers):#取绝对值后求和
    return sum(x for x in numbers)


def readpar(path):
    data_dict = {}
    with open(path, 'r') as file:
    # 逐行读取文件内容
        for line in file:
            # 按照分号分隔键值对
            key, value = line.strip().split(';')

            # 如果值是一个列表，使用eval()函数将其转换为实际的列表对象
            if key == 'listlaterflow' or 'u' or 'q':
                value = eval(value)
            else:
                value = int(value)  # 将非列表的值转换为整数

            # 将键值对存储到字典中
            data_dict[key] = value
    file.close()
    return data_dict
def get_sign(num):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0
if __name__ == "__main__":
    
    
    if os.path.exists('runswmm'):
        shutil.rmtree('runswmm')

# 创建新的文件夹
    os.makedirs('runswmm')
    pathobs = 'obstest/202007'
    for kkkk in range(2,3): 
        pathflow = f'{pathobs}/flow.xlsx'
        path = f'{pathobs}/obs.xlsx'
        pathsheet = 'obs'
        pathsheet2 = 'flow'
        pathpar = f'{pathobs}/par.txt'
        file_content = readpar(pathpar)
        Qk = []
        hottime = int (file_content['hottime'])
        simtime = file_content['simtime']
        stabletime = file_content['stabletime']
        waitingtime = file_content['waitingtime']
        _Maning = [0.01]*13
        listnum = file_content['listlaterflow']
        n = file_content['n']
        u = file_content['u']
        q = file_content['q']
        lentime = file_content['lentime']
        RTStime = int(file_content['RTStime']*4)
        numflowava = int(file_content['numflowava'])#同化流量的时刻数量
        numheadava = int(file_content['numheadava'])#同化用的观测水位的时刻数量
        numdeleteheadb = int(file_content['numdeleteheadb'])
        numdeleteheadf = int(file_content['numdeleteheadf'])
        backdt = int(file_content['RTStime'])
        dq = float(file_content['dq'])
        hdelete = file_content['Iddetet']
        option = file_content['option']
        outpath = 'sim2'
        uu = [[] for x in range(1+len(listnum))]
        
        for i1 in range(len(uu)):  
            uu[i1] = [0 for x in range(1+len(listnum))]
            if i1 == 0:
                uu[i1][i1] = u[0]**2
            else:
                uu[i1][i1] = u[listnum[i1-1]+1]**2
            
        Gkm = [0 for x in range(1+len(listnum))]
        Gku = [0 for x in range(1+len(listnum))]
        Q_rts = [[] for x in range(1+len(listnum))]
        RTS_Gk = []
        # print(dq)
        data = pd.read_excel(
                io = path,
                sheet_name = pathsheet)
        dataflow = pd.read_excel(
                io = pathflow,
                sheet_name = pathsheet2)
        flowhost = dataflow.iloc[:,9].values.tolist() #历史预热期流量，采用历史值，而不是固定值
        Zhost = dataflow.iloc[:,8].values.tolist()
        outflow = dataflow.iloc[:,10].values.tolist() #水库下泄历史流量
        Zinit = []
        for j in range(-int(hottime*4),1,1):
            Zinit.append(dataflow.iloc[j].values[:-2])
        host=[data.iloc[:, 10].values,data.iloc[:, 9].values]#初始流量，初始水位
        if len(Zinit) != 0:
            initZ = Zinit[0]
        else:
            initZ = data.iloc[0, :].values[1:-1]
        host = [flowhost[-int(hottime*4):],Zhost[-int(hottime*4):],outflow[-int(hottime*4):]]
        print('55555',host[0][0])
        print('initZ',initZ)
        lateralflow = [[0 for i in range(int((stabletime + j*waitingtime)*4))] for j in range(8)]
        obsold = copy.copy(data.iloc[0, :])
        
        reset_file("sim/sim"+str(kkkk)+".txt")
        reset_file("sim/simRTS"+str(kkkk)+".txt")
        reset_file("sim/simMean"+".txt")
        reset_file('sim/simz'+str(kkkk)+'.csv')
        reset_file('sim/simz_'+str(kkkk)+'.csv')
        reset_file('sim/simq'+str(kkkk)+'.csv')
        reset_file('sim/simqRTS'+str(kkkk)+'.csv')
        reset_file('sim/simq'+str(kkkk)+'.txt')
        reset_file('sim/dataouthead.csv')
        reset_file('sim/dataoutflow.csv')
        reset_file('sim/time.txt')
        obsitmulti = []
    
        RTS_Cpy = [] # 系统参数协方差(更新后）
        RTS_Cpy_ = [] # 系统参数协方差（更新前）
        RTS_P = [] # 参数 （更新后）
        RTS_P_ = [] # 参数 （更新前）
        RTS_Pp = [] # 参数 （平滑后）
        RTSX_ = []
        RTS_X = []
        # RTS_kp_
        RTS_Q = []
        RTS_Gk = 0 #向后卡尔曼增益
        Gkadam = [0 for x in range(1+len(listnum))]
        RTS_GkL = []
        RTSm = 0
        RTSu = 0
        #卡尔曼平滑
        qt = []
        qstorage = [[] for i in range(numflowava)] #用于每次同化的流量结果，然后取平均值
        
        try:
    # 试图访问变量
                indexheadmean
        except NameError:
    # 如果变量不存在，则创建并初始化变量
                indexheadmean = []
        for i in range(int(len(data)-lentime*4)):
            loop = i
            print(i)
            obsitmulti = []    
            if i == 2800:
                break
#---------------------------observation 
            if len(Zinit) != 0:#第一行作为热启动状态输入
                for j in range(i,i+int((stabletime-hottime)*4)+int(lentime*4)+1+2+int(waitingtime*4*8)+10):#利用多个时间尺度的水位数据同化，获得过去的流量
                    obsitmulti.append(data.iloc[j].values)
                for j in range(len(obsitmulti)):
                    obs = obsitmulti[j]
                    nan_indices = find_nan_indices(obs)
                    if len(nan_indices) == 0:
                       obsold = copy.copy(obsitmulti[j]) 
                    if len(nan_indices) != 0:
                        for x in nan_indices:
                            obsitmulti[j][x] = obsold[x]   
                obsold = copy.copy(obsitmulti[0])
                obsit3 = []
                for t in range(int(lentime*4+1)):
                    ti = (t+1)
                    obsit1 = [obsitmulti[k][j] for k in range(int((stabletime-hottime)*4) + ti ,int((stabletime-hottime)*4 + ti +1)) for j in [1]]
                    obsit2 = [obsitmulti[k][j] for k in range(int((stabletime-hottime)*4) + ti + int(waitingtime*4) ,int((stabletime-hottime)*4 + ti +1 + int(waitingtime*4))) for j in [2,3,4,5,6,7,8]]
                    obsit3.append(obsit1+obsit2) 
                obsit = [obsit3[k][j] for k in range(len(obsit3)) for j in range(len(obsit3[0]))]
                Zend = [obsitmulti[k][-2] for k in range(len(obsitmulti))]
                outflowend = [obsitmulti[k][0] for k in range(len(obsitmulti))]
                inflow = [obsitmulti[k][-1] for k in range(len(obsitmulti))]
                
                if i == 0:
                    host[1] = host[1][:]+Zend
                    host[2] = host[2][:]+outflowend
                    q[0] = host[0][-1]
                if i >0:
                    host[1] = host[1][1:]+[Zend[-1]]
                    host[2] = host[2][1:]+[outflowend[-1]]
                if i > 0:
                    q[0] = meanq
                    print(meanq)
                    for i5 in range(len(q)):
                        if q[i5]<0:
                            q[i5] = 0
                numqt = numflowava
                qt = inflow[0:numqt] #四个时刻的流量
                # print(q)
#---------------------------observation               
#---------------------------KalmanFilter               
                enkf = KalmanFilter(obsit,host,initZ,lateralflow,lentime)
                enkf.num_cores = file_content['numcores'] #并行核数
                if i >=1:
                    enkf.hotoption1 = 0
                enkf.time = stabletime 
                enkf.simtime = simtime
                enkf.waitingtime = waitingtime
                enkf.listnum = listnum
                enkf.hottime = hottime
                enkf.maxflow = obsitmulti[0][0]#水库下泄流量当前值 # 已经取消
                enkf.inflow = inflow[1:]
                enkf.qt = qt
                enkf.obsitmulti = obsitmulti
                enkf.nha = numheadava
                enkf.num = numdeleteheadb #从近坝开始
                enkf.numb = numdeleteheadf #从近上边界开始
                enkf.D1_prediction(q, _Maning , u, n)
                enkf.observation()
                Galf = enkf.calculate_average_difference()
                print('Galf',Galf)
                Zinit.append(obsitmulti[0][1:-1])   
                enkf.K()
                _Z,_Q,Z_ = enkf.update()
#---------------------------KalmanFilter 
                
                q = _Q
                for kk in range(len(qstorage)):
                    qstorage[kk].append(_Q[kk])
                zlnum = 0 #标记支流位置
                _Qtemp = np.array([_Q[:numflowava]])#暂时存储干流流量
                for j in range(len(lateralflow)):
                    if j in listnum:
                        zlnum += 1
                        _Qtemp = np.append(_Qtemp,_Q[numflowava-1 + zlnum]) # 将包含的支流加入
                    else:
                        _Qtemp = np.append(_Qtemp,0) # 不包含的支流流量为零                        
#--------------------------------------支流-支流预热+支流模拟
                _Q = _Qtemp  #替换
                q = _Qtemp  #替换
                print('_Q',[int(x) for x in _Q])
                print('qstorage',int(sum(qstorage[0])/len(qstorage[0])),host[0][-1])
                                
                qnest = int(sum(qstorage[0])/len(qstorage[0]))   
                qnest = copy.copy(int(qnest))    
                host[0] = host[0][1:]+[qnest]
                meanq = qnest
                for j in range(len(lateralflow)):
                    lateralflow[j]  = lateralflow[j][1:] + [_Q[numflowava + j]]
#-------------------------------------支流  end
#-------------------------------------更新初始水位
                # if len(Zinit) != 0:
                #     del Zinit[0]
                #     initZ = copy.copy(Zinit[0])
                # else:
                #   initZ = copy.copy(obsitmulti[0][1:-1])
                # print('initZ',initZ)
                # print(enkf.kp,enkf.kx)
#-------------------------------------更新初始水位
                b1 = 0.85
                b2 = 0.85
                for i5 in range(1,len(Q_rts)):
                    Q_rts[i5].append(lateralflow[listnum[i5-1]][-1] - lateralflow[listnum[i5-1]][-2])
                    if len(Q_rts[i5]) >= 2:
                        Gkm[i5] =b1*abs(Gkm[i5]) + (1-b1)*(Q_rts[i5][-1])*get_sign(Q_rts[i5][-2])
                        Gku[i5] =b2*Gku[i5] + (1-b2)*abs((Q_rts[i5][-1]))
                Q_rts[0].append(host[0][-1]-host[0][-2])
                if len(Q_rts[0]) >= 2:
                    Gkm[0] =b1*abs(Gkm[0]) + (1-b1)*(Q_rts[0][-1])*get_sign(Q_rts[0][-2])
                    Gku[0] =b2*Gku[0] + (1-b2)*abs((Q_rts[0][-1]))
                    
                for i5 in range(0,len(Q_rts)):
                    if len(Q_rts[i5]) >= 2: 
                        Gkadam[i5] = abs(Gkm[i5])/(Gku[i5]+1)
                if len(Q_rts[i5]) >= 2:
                    RTS_GkL.append(Gkadam)
                print(Gkm,Gku)
                print(Gkadam)
#-------------------------------------save result
                
                
                append_data_to_txt("sim/sim"+str(kkkk)+".txt",[int(sum(_Q))])
                append_data_to_txt("sim/simMean"+".txt",[qnest])                
                for kk in range(len(qstorage)):
                    if kk < len(qstorage)-1:
                        qstorage[kk] = qstorage[kk+1]
                    else:
                        qstorage[kk] = []
                        
                row = [round(x,3) for x in _Z]
                rowq = [round(x,2) for x in _Q] + [sum_of_absolute_values(_Q)]
                rowz_ = [round(x,3) for x in Z_]
                
                with open('sim/simq'+str(kkkk)+'.csv', mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(rowq)
                append_data_to_txt("sim/simq"+str(kkkk)+".txt",[rowq[-1]])
                with open('sim/simz'+str(kkkk)+'.csv', mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['_sim']+row)
                    writer.writerow(['obs']+enkf.obs)
                    
                with open('sim/simz_'+str(kkkk)+'.csv', mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['sim_']+rowz_)
                    writer.writerow(['obs']+row) 
#--------------------------------------save result                 
#--------------------------------------Begin of RTS       
            RTS_Cpy.append(enkf._Cpp)
            RTS_Cpy_.append(enkf.Cpp_)
            RTS_P.append(enkf._p)
            RTS_P_.append(enkf.p)
            RTSX_.append(enkf.x_)
            RTS_X.append(enkf._x)
            rts_t = RTStime
                    
            if i >= 2 and option == True: #平滑时间窗
                RTS_Pp=[RTS_P[-1]]
                
                
                
                for i1 in range(int(len(RTS_P))-2,-1,-1):              
                    RTS_Gk = np.dot(RTS_Cpy[i1],np.linalg.pinv(RTS_Cpy_[i1+1]))
                    # RTS_Gk = np.dot(RTS_Cpy[i1],np.linalg.pinv(RTS_Cpy[i1]+uu))
                  
                    if i1 == int(len(RTS_P))-2:
                        print('RTS_Gk',RTS_Gk)
                    for i3 in range(len(RTS_Gk)):
                        for i4 in range(len(RTS_Gk[0])):
                            if i3 == i4:
                                RTS_Gk[i3][i4] = min(0.98,RTS_Gk[i3][i4])
                                # if i < 5:
                                # RTS_Gk[i3][i4] = min(RTS_GkL[i1][i3]*0.40 + 0.60,0.98)
                                # RTS_Gk[i3][i4] = min(RTS_GkL[i1][i3]*Galf,0.99)
                                # else:
                                    # RTS_Gk[i3][i4] = RTS_GkL[i1][i3]*0.4 + 0.598
                                pass
                            else:
                                RTS_Gk[i3][i4] = 0
                    # RTS_Gk[0][0] =0.99
                    # RTS_Gk[0][0] =0.9*RTSm_/(RTSu_**0.5+0.000001)
                    if i1 == int(len(RTS_P))-2:
                        print(RTS_Gk)
                    RTSp = (RTS_P[i1].T+np.dot(RTS_Gk,(RTS_Pp[-1].T-RTS_P_[i1+1].T))).T  
                    RTS_Pp.append(RTSp)   
                RTS_Q = [] # for main stream
                RTS_lQ = [[] for i3 in range(len(lateralflow))] # for tributaries               
                for i3 in range(int(len(RTS_Pp))-1,-1,-1):                    
                    Q = np.mean(RTS_Pp[i3],axis=0)
                    zlnum = 0 #标记
                    _Qtemp = np.array([Q[:numflowava]]) #暂时存储
                    for j in range(len(lateralflow)):
                        if j in listnum:
                            zlnum += 1
                            _Qtemp = np.append(_Qtemp,Q[numflowava-1+zlnum])
                        else:
                            _Qtemp = np.append(_Qtemp,0)# 去除不包含的支流
                    Q = _Qtemp  #替换
                    RTS_Q.append(int(Q[0]))
                    # print('RTSQ',Q)
                    for i4 in range(len(RTS_lQ)):
                        RTS_lQ[i4].append(Q[numflowava+i4])
                RTSt = RTStime
                kkk = 0
        #-------------alter historial flow with smoothing flow
                # print(len(RTS_Q),RTS_lQ[2])              
                for kk in range(len(RTS_Q)):
                    if len(RTS_Q) > RTSt:
                        kkk = int(len(RTS_Q)-RTSt)
                        if RTS_Q[kk+kkk] < 0:
                            RTS_Q[kk+kkk] = 0
                        host[0][-len(RTS_Q)+kk+kkk] = copy.copy(RTS_Q[kk+kkk])
                        if kk + kkk == len(RTS_Q)-1:
                            break
                    else: 
                        host[0][-len(RTS_Q)+kk] = copy.copy(RTS_Q[kk])
                for i3 in range(len(lateralflow)):        
                    for kk in range(len(RTS_lQ[i3])):
                        if len(RTS_lQ[i3]) > RTSt:
                            kkk = int(len(RTS_lQ[i3])-RTSt)
                            if RTS_lQ[i3][kk+kkk] < 0:
                                RTS_lQ[i3][kk+kkk] = 0
                            lateralflow[i3][-len(RTS_lQ[i3])+kk+kkk] = copy.copy(RTS_lQ[i3][kk+kkk])
                            if kk + kkk == len(RTS_lQ[i3])-1:
                                break
                        else: 
                            lateralflow[i3][-len(RTS_lQ[i3])+kk] = copy.copy(RTS_lQ[i3][kk])
                
        #-------------alter historial flow
                                                
                if i >= RTStime:
                    RTSqr = copy.copy(RTS_Q[-RTStime])
                    rowqRTS = [int(RTSqr)]+[lateralflow[i3][-RTStime] for i3 in range(len(lateralflow))]        
                    append_data_to_txt("sim/simRTS"+str(kkkk)+".txt",[int(RTSqr)])
                    with open('sim/simqRTS'+str(kkkk)+'.csv', mode='a', newline='', encoding='utf-8') as csv_fileRTS:
                        writer = csv.writer(csv_fileRTS)
                        writer.writerow(rowqRTS)                
                if len(RTS_Q) > RTSt:
                    # print(host[0][-RTSt:])
                    print(RTS_Q[-RTSt:])
                    print(lateralflow[4][-RTSt:])             
            meanq = host[0][-1]
