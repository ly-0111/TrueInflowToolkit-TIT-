import numpy as np
import os
import pandas as pd
import shutil
import multiprocessing
import psutil
def HD(Q_,Q,Z_,Z,time):
    # print(os.getcwd())
    os.chdir('./D1model')
    array = list(range(17))
    with open('Inputfile/BOUNDARY_Q.txt', 'w') as f:#修改上游流量边界文件
        f.write('17\n')
        for id in array:
            if id>11:
                f.write(f'{id}\t{Q}\n')
            else:
                f.write(f'{id}\t{Q_}\n')
    f.close()
    with open('Inputfile/BOUNDARY_Z.txt', 'w') as f:#修改下游水位边界文件
        f.write('17\n')
        for id in array:
            if id>11: 
                f.write(f'{id}\t{Z}\n')
            else:
                f.write(f'{id}\t{Z_}\n')
    f.close()
    os.system('.\main.exe')
    df = pd.read_csv('Outputfile/DynamicRESULT.csv')
    for index, row in df.iterrows():
        # print(row[0])
        if row[0] == time:
            # print(row[0])
            break
    row = [row[i] for i in range(len(row)) if i%2 !=0]
    # print(row)
    os.chdir('../')
    return np.array(row)
# HD(2000,150,1.0)

def recreate_txt_file(file_path):
    """
    检查文件是否存在，存在则删除并重新创建一个空的txt文件。

    参数:
    file_path (str): 要检查和重新创建的文件的路径。
    """
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 删除文件
        os.remove(file_path)
        # print(f"文件 '{file_path}' 已存在，已删除。")
    
    # 创建新的空文件
    with open(file_path, 'w') as file:
        pass  # 创建一个空文件
    
    # print(f"文件 '{file_path}' 已重新创建。")
def checkdir(path):
    # 检查文件夹是否存在
    if os.path.isdir(path):
        # 如果文件夹存在，则删除
        shutil.rmtree(path)
        print("文件夹已删除。")
    else:
        # 如果文件夹不存在，打印消息
        print("文件夹不存在。")

def append_data_to_txt(file_path, data):
    """
    向指定的txt文件中逐行追加数据。

    参数:
    file_path (str): txt文件的路径。
    data (list of str): 要追加的数据列表，每个元素为一行。
    """
    with open(file_path, 'a') as file:
        for line in data:
            file.write(str(line) + '\n')  # 在每行末尾添加换行符

def run(ID,core,Q_=0,Q=0,Z_=0,Z=0,time=0):
    # print(os.path.isdir(f'./{j}'))
    process_id = multiprocessing.current_process().pid
    # 通过进程ID获取进程对象
    process = psutil.Process(process_id)
    # 获取进程所在的CPU核心
    
    process.cpu_affinity([ID%core])
    
    cpu_affinity = process.cpu_affinity()[0]

    if os.path.isdir(f'./run/{cpu_affinity}') == False:
        shutil.copytree('./D1model', f'./run/{cpu_affinity}')
    os.chdir(f'./run/{cpu_affinity}')
    
    array = list(range(13))
    with open('Inputfile/BOUNDARY_Q.txt', 'w') as f:#修改上游流量边界文件
        f.write('13\n')
        for id in array:
            if id>7:
                f.write(f'{id}\t{Q}\n')
            else:
                f.write(f'{id}\t{Q}\n')
    f.close()
    with open('Inputfile/BOUNDARY_Z.txt', 'w') as f:#修改下游水位边界文件
        f.write('13\n')
        for id in array:
            if id>7: 
                f.write(f'{id}\t{Z_}\n')
            else:
                f.write(f'{id}\t{Z_}\n')
    f.close()
    array = list(range(9))
    with open('Inputfile/iniZ.txt', 'w') as f:#修改初始水位边界文件，作为状态转移
        f.write('9\n')
        for id in array:
            f.write(f'{id}\t{Z[id]}\n')
            
    f.close()
    os.system('.\main.exe')
    df = pd.read_csv('Outputfile/DynamicRESULT.csv')
    for index, row in df.iterrows():
        # print(row[0])
        if row[0] == time:
            # print(row[0])
            break
    row = [row[i] for i in range(len(row)) if i%2 !=0]
    # print(row)
    os.chdir('../../')
    return (str(ID),[Q,np.array(row)])
# def worker(j):
#     # 获取当前进程的ID
#     process_id = multiprocessing.current_process().pid
#     # 通过进程ID获取进程对象
#     process = psutil.Process(process_id)
#     # 获取进程所在的CPU核心
#     process.cpu_affinity([0])
#     # 获取进程所在的 CPU 核心
#     cpu_affinity = process.cpu_affinity()
# #     print(f"进程 {process_id} 运行在 CPU 核心 {cpu_affinity} 上。")

# def main():
#     with multiprocessing.Pool() as pool:
#         pool.map(worker, range(4))  # 假设有4个任务
# if __name__ == "__main__":
#     main()