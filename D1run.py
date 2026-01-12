import chardet
from pyswmm import Simulation,Nodes,Links,SystemStats,RainGages,Output,NodeSeries,LinkSeries
import os
import numpy as np
import pandas as pd
from D1 import HD,recreate_txt_file,append_data_to_txt,run
import time,copy
import multiprocessing
import psutil
import shutil
import datetime
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']



def D1run(ID, core,Hotoption,Hotoption1,hottime,Qrandom, Qlist = [], Zlist = [],ZL = [163, 155, 154, 153, 153,153,153,153,153], sub_laterQ_lists = [],time = 0,lentime = 0,waitingtime = 0,listnum=[],maxflow = 0,dt = 4):
    
    process_id = multiprocessing.current_process().pid
    # 通过进程ID获取进程对象
    process = psutil.Process(process_id)
    # 获取进程所在的CPU核心
    available_cpus = list(range(psutil.cpu_count()))
    # process.cpu_affinity([ID%core])
    process.cpu_affinity([available_cpus[ID%core]])
    cpu_affinity = process.cpu_affinity()[0]
    # print(ID)
    # print(ID%core)
    # exit()
    if os.path.isdir(f'./runswmm/{cpu_affinity}') == False:
        shutil.copytree('./all', f'./runswmm/{cpu_affinity}')
    os.chdir(f'./runswmm/{cpu_affinity}')
    
    with open('88.inp', 'r') as file:
        lines = file.readlines()
    file.close()
    
    
    NC = 0 #使用全断面模型
    # NC = 55 #使用delete文件夹中的模型
    man = [0.022,0.022,0.023,0.024,0.022,0.022,0.022,0.022,0.025,0.0223]
    man = [0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
    #man = [0.0232,0.045,0.029,0.032,0.035,0.032,0.067,0.069,0.066,0.049]
    #man = [0.032,0.032,0.033,0.034,0.032,0.032,0.032,0.032,0.035,0.0323]
    for j in range(len(lines)):
        
        l = lines[j].split()
        if len(l) != 0:
            
            if l[0] == 'NC':
                NC += 1  
                if NC <= 34:
                    lines[j] = 'NC' + '\t' + str(man[0]) + '\t' + str(man[0]) + '\t' + str(man[0]) + '\n'
                if 34 < NC <= 63:
                    m = man[0] + (man[1]-man[0])*(NC-34)/(63-34)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 63 < NC <= 86:
                    m = man[1] + (man[2]-man[1])*(NC-63)/(86-63)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if  86 < NC <= 130:
                    m = man[2] + (man[3]-man[2])*(NC-86)/(130-86)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 130 < NC <= 154:
                    m = man[3]+ (man[4]-man[3])*(NC-130)/(154-130)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 154 < NC <= 184:
                    m = man[4]+ (man[5]-man[4])*(NC-154)/(184-154)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 184 < NC <= 212:
                    m = man[5]+ (man[6]-man[5])*(NC-184)/(212-184)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 212 < NC <= 231:
                    m = man[6]+ (man[7]-man[6])*(NC-212)/(231-212)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 231 < NC <= 260:
                    m = man[7]+ (man[8]-man[7])*(NC-231)/(260-231)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                if 260 < NC <= 288:
                    m = man[8]+ (man[9]-man[8])*(NC-260)/(288-260)
                    lines[j] = 'NC' + '\t' + str(m) + '\t' + str(m) + '\t' + str(m) + '\n'
                    
            if l[0] == '[JUNCTIONS]':
                # for i in range(j+3 , j+2+288-1):
                #     lj = lines[i].split()
                #     if int(lj[0]) <= 51:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[0]+  - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 51< int(lj[0]) <= 86:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[1]+(ZL[0]-ZL[1])*(86-int(lj[0]))/(86-51) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 86<int(lj[0]) <= 130:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[2]+(ZL[1]-ZL[2])*(130-int(lj[0]))/(130-86) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 130<int(lj[0]) <= 154:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[3]+(ZL[2]-ZL[3])*(154-int(lj[0]))/(154-130) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 154<int(lj[0]) <= 184:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[4]+(ZL[3]-ZL[4])*(184-int(lj[0]))/(184-154) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 184<int(lj[0]) <= 212:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[5]+(ZL[4]-ZL[5])*(212-int(lj[0]))/(212-184)- float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 212<int(lj[0]) <= 231:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[6]+(ZL[5]-ZL[6])*(231-int(lj[0]))/(231-212) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 231<int(lj[0]) <= 260:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[7]+(ZL[6]-ZL[7])*(260-int(lj[0]))/(260-231) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                #     elif 260<int(lj[0]) <= 288:
                #         lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[8]+(ZL[7]-ZL[8])*(287-int(lj[0]))/(287-260) - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                for i in range(j+3 , j+2+288-1-56):
                    lj = lines[i].split()
                    if int(lj[0]) <= 86:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[0] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 86<int(lj[0]) <= 130:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[1] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 130<int(lj[0]) <= 154:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[2] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 154<int(lj[0]) <= 184:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[3] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 184<int(lj[0]) <= 212:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[4]- float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 212<int(lj[0]) <= 231:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[5] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 231<int(lj[0]) <= 260:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[6] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    elif 260<int(lj[0]) <= 288:
                        lines[i] = str(lj[0] + '\t\t\t'+ lj[1] + '\t\t' + lj[2] + '\t\t' + str(round(ZL[7] - float(lj[1]),1)) + '\t\t' +  lj[4] + '\t\t' +  lj[5]) + '\n'
                    
                    if l[0] == '[OUTFALLS]' :
                            break  
            if l[0] == '[TIMESERIES]':
                N_1 = 0
                N_2 = 0
                N_33 = 0;N_44 = 0;N_55 = 0;N_66 = 0;N_77 = 0;N_88 = 0;N_99 = 0;N_100 = 0
                for i in range(j+3 , len(lines)):
                    
                    lj = lines[i].split()
                    if len(lj) != 0:
                        if lj[0] == '1' :
                            N_1 += 1 
                            if N_1 <= len(Zlist):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(Zlist[N_1-1]) + '\n'
                            
                        if lj[0] == '22' :
                            N_2 += 1
                            if N_2 <= len(Qlist):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(Qlist[N_2-1]) + '\n'
                        if lj[0] == '33' :
                            N_33 += 1
                            if N_33 <= len(sub_laterQ_lists[0]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[0][N_33-1]) + '\n'
                        if lj[0] == '44' :
                            N_44 += 1
                            if N_44 <= len(sub_laterQ_lists[1]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[1][N_44-1]) + '\n'
                        if lj[0] == '55' :
                            N_55 += 1
                            if N_55 <= len(sub_laterQ_lists[2]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[2][N_55-1]) + '\n'
                        if lj[0] == '66' :
                            N_66 += 1
                            if N_66 <= len(sub_laterQ_lists[3]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[3][N_66-1]) + '\n'
                        if lj[0] == '77' :
                            N_77 += 1
                            if N_77 <= len(sub_laterQ_lists[4]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[4][N_77-1]) + '\n'
                        if lj[0] == '88' :
                            N_88 += 1
                            if N_88 <= len(sub_laterQ_lists[5]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[5][N_88-1]) + '\n'
                        if lj[0] == '99' :
                            N_99 += 1
                            if N_99 <= len(sub_laterQ_lists[6]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[6][N_99-1]) + '\n'
                        if lj[0] == '100' :
                            N_100 += 1
                            if N_100 <= len(sub_laterQ_lists[7]):
                                lines[i] = lj[0] +'\t\t'+ lj[1] +'\t\t'+ lj[2] +'\t\t'+ str(sub_laterQ_lists[7][N_100-1]) + '\n'
                        if l[0] == '[PATTERNS]' :
                            break
                
    with open('287.inp', 'w') as file:
        file.writelines(lines)
    file.close()
    
    hotoption = copy.copy(Hotoption)
    hotoption1 = copy.copy(Hotoption1)
    
    with Simulation('287.inp') as sim:
        system_state = SystemStats(sim)
        stepnum = 0
        head = []
        head1 = []
        head2 = []
        head3 =[]
        # IDjunc = [34,51,63,86,130,154,184,212,231,260,286] #采用全部观测断面
        # IDjunc1 = [51]
        IDjunc2 = [86,130,154,184,212,231,260,286,289] #采用8个观测断面
        # IDjunc2 = [79,130,155,189,212,231,260,286] #采用8个观测断面

        current_time = 0
        time2 = time
        stepnum = 0
        num = 0
        mn = 0
        num2 = 0
        num1 = 0
        

        if hotoption == 0 and hotoption1 == 1:
            sim.use_hotstart('hotstart.HSF')
            sim.start_time = datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=hottime-dt)
            sim.end_time = datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=hottime+int(lentime+waitingtime*8+3))
            num = int((hottime-dt)/0.25)
            num2 = 0
            num1= num
            num = 0
        if hotoption1 ==0 :
            sim.use_hotstart('hotstart.HSF')
            if hotoption == 1:
                sim.start_time = datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=hottime-0.25-dt)
            if hotoption == 0:
                sim.start_time = datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=int(hottime-dt))
            sim.end_time = datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=hottime+int(lentime+waitingtime*8+3))
            if hotoption == 1:
                num = int((hottime-dt-0.25)/0.25)
            if hotoption == 0:
                num = int((hottime-dt)/0.25)
            num2 = 0
            num1= num

            num = 0

        starttime = sim.start_time
        for step in sim:
            current_time = sim.current_time 
            if hotoption1 == 0 and current_time >= datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=int(hottime-dt)) and hotoption == 1:
                # print('sim.save_hotstart','xxxxxxxxxxxxxxxxxxxxxxxxxx')
                sim.save_hotstart('hotstart.HSF')               
                hotoption1 = 2                
            if stepnum == 0 and hotoption1 == 1:                
                Links(sim)['290'].target_setting = 0.000001
                stepnum += 5
            if current_time > sim.start_time + datetime.timedelta(hours=0.25*num):                
                num += 1
                num1 += 1
                if num1 >= len(maxflow):
                    num1 = len(maxflow)-1
                if num1 > 40*(hotoption1):
                    # print(num1)
                    Links(sim)['287'].flow_limit = int(maxflow[num1])# whether pump
                    Links(sim)['286'].flow_limit = int(maxflow[num1])# whether pump
                    pass
            if current_time > sim.start_time + datetime.timedelta(hours=0.25*num2) and num1 > 40*(hotoption1):
                num2 +=1
                dq = (int(maxflow[num1])-Links(sim)['287'].flow)
                if dq < 0:
                    dq = 0
                Links(sim)['290'].target_setting = (dq/30000+0.000001) # whether pump
            if current_time >= sim.start_time+datetime.timedelta(hours=hottime-dt) and hotoption == 1 and hotoption1==1:
                sim.save_hotstart('hotstart.HSF')
                print('savehotstart',sim.current_time)
                hotoption = 0    
            pass
    
    numhead_start = (datetime.datetime(2022, 10, 1, 00, 00)+datetime.timedelta(hours=time) - starttime).total_seconds() / (60*15)
    # print(numhead_start) 
    # if hotoption == 1:
        # numhead_start = numhead_start+1
    with Output('287.out') as out:

        # flow = [list(LinkSeries(out)[str(180)].flow_rate.values())]
        flow = [list(NodeSeries(out)[str(169)].lateral_inflow.values())]
        
        head11 = [list(NodeSeries(out)[str(x)].hydraulic_head.values())[int(numhead_start):] for x in IDjunc2]
        # head11 = [list(NodeSeries(out)[str(x)].hydraulic_head.values())[:] for x in IDjunc2]
        for kk in range(len(head11)):
            head11[kk] = [round(x,4) for x in head11[kk]]
    headmulti = [head[k][j] for k in range(len(head)) for j in range(len(head[0]))]
    sim.close()
    os.chdir('../../')
    # Q = Qrandom+[first_non_zero(sub_laterQ_lists[x][-1]) for x in listnum]#不包含支流
    Q = Qrandom+[sub_laterQ_lists[x][-1] for x in listnum]#不包含支流
    return (str(ID),[np.array(Q),np.array(headmulti),current_time,np.array(flow),np.array(head11)])

def first_non_zero(lst):
    for num in lst:
        if num != 0:
            return num
    return 0  # 如果所有元素都是零或列表为空
def reset_file(file_path):
    """
    检查文件是否存在，存在则删除，然后重新创建一个同名的空文件。
    不存在则直接创建一个空文件。
    
    :param file_path: 文件路径
    """
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，则删除文件
        os.remove(file_path)
        # print(f"文件 '{file_path}' 已存在并被删除。")
    else:
        print(f"文件 '{file_path}' 不存在，将创建一个新文件。")
    
    # 创建一个同名的空文件
    with open(file_path, 'w', encoding='utf-8') as file:
        pass  # 创建一个空文件
    
    # print(f"文件 '{file_path}' 已成功创建。")
def ff_d(seq1, seq2, dq):
    # 找出第一个不同的索引
   
    for index, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b and abs(a-b) > dq:
            return [index,a-b]
    # 如果完全相同，返回 None
    return [None,None]
def add_column_to_csv(input_file, output_file, column_name, new_column_data, fill_value=None):
    """
    向 CSV 文件中添加一列新数据。
    
    参数：
    input_file (str): 输入的 CSV 文件名。
    output_file (str): 输出的 CSV 文件名。
    column_name (str): 新列的列名。
    new_column_data (list): 新列的数据。
    fill_value: 用于填充不足部分的值。
    """
    try:
        # 尝试读取现有的 CSV 文件
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        # 如果文件为空，创建一个新的 DataFrame
        df = pd.DataFrame({column_name: new_column_data})

    # 如果新列数据的长度不足，用 fill_value 进行填充
    if len(new_column_data) < len(df):
        new_column_data.extend([fill_value] * (len(df) - len(new_column_data)))
    # 如果新列数据的长度超出，截断多余部分
    elif len(new_column_data) > len(df):
        new_column_data = new_column_data[:len(df)]

    # 如果原 DataFrame 为空，根据新列数据的长度创建索引
    if df.empty:
        df = pd.DataFrame({column_name: new_column_data})
    else:
        # 添加新列到已有 DataFrame
        df[column_name] = new_column_data

    # 保存到新的 CSV 文件
    df.to_csv(output_file, index=False)

    print("数据已成功添加到新的文件中。")
def head_thred(matrix):
    # 将绝对值大于 0.1 的元素替换为 0.1 或 -0.1
    clipped_matrix = np.where(matrix > 0.05, 0.05, matrix)
    clipped_matrix = np.where(clipped_matrix < -0.05, -0.05, clipped_matrix)
    return clipped_matrix
def flow_thred(matrix,ha,listnum):
    # 将绝对值大于 0.1 的元素替换为 0.1 或 -0.1
    # clipped_matrix = np.where(matrix > 0.1, 0.1, matrix)
    # print(ha)
    clipped_matrix = np.where(matrix < 0, 0, matrix)
    # for k in range(1,len(clipped_matrix)):
    #     for j in range(len(clipped_matrix[0])):
    #         if ha[int(j % len(ha))] < listnum[k-1]:
    #             clipped_matrix[k][j] = 0
    return clipped_matrix


