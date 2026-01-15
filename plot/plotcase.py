import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import configparser
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates  # 用于日期格式化和刻度管理
def mainflow(a=2,b=5,c=(0.00, 0.75, 0.75)):
    # 读取 Excel 文件
    file_path = 'plot.xlsx'  # 替换为你的文件路径
    data = pd.read_excel(file_path,sheet_name='Section4.2.3')
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # 提取数据
    time = pd.to_datetime(data.iloc[:, 1])  # 第一列为时间，转换为 datetime 格式
    method1 = data.iloc[:, a]  # 第二列为 method1 的结果
    method2 = data.iloc[:, b]  # 第三列为 method2 的结果
    difference = method1 - method2  # 计算差值

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 右轴：差值的柱状图（先绘制柱状图）
    # custom_orange = (0.22, 0.49, 0.72)
    # custom_orange = '#C4D870'
    # custom_orange = '#EEAD85'
    custom_orange = c
    ax2 = ax1.twinx()
    ax2.set_ylabel('Error (m³/s)', color='#FF0000', fontsize=30)
    bar_width = 0.1  # 调整柱状图宽度
    bars = ax2.bar(time, difference, color=custom_orange, alpha=0.1, width=bar_width, label='Difference', zorder=1)  # zorder=1 表示柱状图在底层
    ax2.tick_params(axis='y', labelcolor='#FF0000', labelsize=30,direction='in')

    # 设置右 y 轴范围和间隔
    ax2.set_ylim(-5000, 5000)  # 设置右轴 y 轴范围为 [-7000, 4000]
    ax2.yaxis.set_major_locator(MultipleLocator(2000))  # 设置主刻度间隔为 2000
    if b != 3:
        # 左轴：method1 和 method2 的曲线图（后绘制曲线图）
        ax1.set_ylabel('Inflow (m³/s)', color='black', fontsize=30)
        # ax1.plot(time, method2, label='WB', color=(0.22, 0.49, 0.72), linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        ax1.plot(time, method1, label='"Ture" inflow', color='black', linestyle='-', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='WB', color=(0.22, 0.49, 0.72), linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='AM', color='#C4D870', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='EnKF', color='#EEAD85', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        ax1.plot(time, method2, label='RTS-320km', color=c, linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='time', color='#FF0000', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        ax1.tick_params(axis='y', labelcolor='black', labelsize=30,direction='in')
        
    else:
        # 左轴：method1 和 method2 的曲线图（后绘制曲线图）
        ax1.set_ylabel('Inflow (m³/s)', color='black', fontsize=30)
        ax1.plot(time, method2, label='WB', color=c, linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        ax1.plot(time, method1, label='"Ture" inflow', color='black', linestyle='-', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='WB', color=(0.22, 0.49, 0.72), linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='AM', color='#C4D870', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='EnKF', color='#EEAD85', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='RTS-320km', color=c, linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        # ax1.plot(time, method2, label='time', color='#FF0000', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
        ax1.tick_params(axis='y', labelcolor='black', labelsize=30,direction='in')
    # 设置左 y 轴范围和间隔
    ax1.set_ylim(19000, 71000)  # 设置左轴 y 轴范围为 [20000, 70000]
    # ax1.set_ylim(1000, 7000)  # 设置左轴 y 轴范围为 [20000, 70000] lateral
    ax1.yaxis.set_major_locator(MultipleLocator(10000))  # 设置主刻度间隔为 10000
    # ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置主刻度间隔为 10000 lateral
    # 设置 x 轴为日期，并控制日期间隔
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))  # 每隔 2 天设置一个主刻度
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式为 年-月-日
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置日期格式为 年-月-日
    plt.xticks(rotation=0,ha='right')  # 旋转日期标签，防止重叠
    # ax1.set_xticklabels(time, rotation=0, ha='right')
    # 设置 x 轴刻度字体大小
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')  # 设置横坐标刻度标签右对齐

    ax1.tick_params(axis='x', labelsize=30,direction='in')  # 设置 x 轴刻度值字体大小为 18
    fig.tight_layout()  # 调整布局避免标签重叠
    # plt.savefig('plot/Figure_WB_3cos0.01.png', dpi=300, bbox_inches='tight')
    plt.show()

def mainflowL(a=2,b=5,c=(0.00, 0.75, 0.75)):
    # 读取 Excel 文件
    file_path = 'plot.xlsx'  # 替换为你的文件路径
    data = pd.read_excel(file_path,sheet_name='Section4.2.2')
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # 提取数据
    time = pd.to_datetime(data.iloc[:, 1])  # 第一列为时间，转换为 datetime 格式
    method1 = data.iloc[:, a]  # 第二列为 method1 的结果
    method2 = data.iloc[:, b]  # 第三列为 method2 的结果
    difference = method1 - method2  # 计算差值

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 右轴：差值的柱状图（先绘制柱状图）
    # custom_orange = (0.22, 0.49, 0.72)
    # custom_orange = '#C4D870'
    # custom_orange = '#EEAD85'
    custom_orange = c
    ax2 = ax1.twinx()
    ax2.set_ylabel('Error (m³/s)', color='#FF0000', fontsize=30)
    bar_width = 0.1  # 调整柱状图宽度
    bars = ax2.bar(time, difference, color=custom_orange, alpha=0.1, width=bar_width, label='Difference', zorder=1)  # zorder=1 表示柱状图在底层
    ax2.tick_params(axis='y', labelcolor='#FF0000', labelsize=30,direction='in')

    # 设置右 y 轴范围和间隔
    ax2.set_ylim(-5000, 5000)  # 设置右轴 y 轴范围为 [-7000, 4000]
    ax2.yaxis.set_major_locator(MultipleLocator(2000))  # 设置主刻度间隔为 2000

    # 左轴：method1 和 method2 的曲线图（后绘制曲线图）
    ax1.set_ylabel('Lateral inflow (m³/s)', color='black', fontsize=30)
    # ax1.plot(time, method2, label='WB', color=(0.22, 0.49, 0.72), linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    ax1.plot(time, method1, label='"Ture" total flow', color='black', linestyle='-', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    # ax1.plot(time, method2, label='WB', color=(0.22, 0.49, 0.72), linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    # ax1.plot(time, method2, label='AM', color='#C4D870', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    # ax1.plot(time, method2, label='EnKF', color='#EEAD85', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    ax1.plot(time, method2, label='RTS', color=c, linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    # ax1.plot(time, method2, label='time', color='#FF0000', linestyle='--', marker=None, linewidth=2, zorder=2)  # zorder=2 表示曲线在上层
    ax1.tick_params(axis='y', labelcolor='black', labelsize=30,direction='in')

    # 设置左 y 轴范围和间隔
    # ax1.set_ylim(19000, 71000)  # 设置左轴 y 轴范围为 [20000, 70000]
    ax1.set_ylim(1000, 7000)  # 设置左轴 y 轴范围为 [20000, 70000] lateral
    # ax1.yaxis.set_major_locator(MultipleLocator(10000))  # 设置主刻度间隔为 10000
    ax1.yaxis.set_major_locator(MultipleLocator(1000))  # 设置主刻度间隔为 10000 lateral
    # 设置 x 轴为日期，并控制日期间隔
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))  # 每隔 2 天设置一个主刻度
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 设置日期格式为 年-月-日
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置日期格式为 年-月-日
    plt.xticks(rotation=0,ha='right')  # 旋转日期标签，防止重叠
    # ax1.set_xticklabels(time, rotation=0, ha='right')
    # 设置 x 轴刻度字体大小
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')  # 设置横坐标刻度标签右对齐

    ax1.tick_params(axis='x', labelsize=30,direction='in')  # 设置 x 轴刻度值字体大小为 18

    # 调整布局
    fig.tight_layout()  # 调整布局避免标签重叠
    # plt.savefig('plot/Figure_WB_3cos0.01.png', dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    # 可以指定不同的配置文件
    mainflow(a=2,b=3,c=(0.22, 0.49, 0.72))
    mainflow(a=2,b=4,c='#C4D870')
    mainflow(a=2,b=6,c='#EEAD85')
    mainflow(a=2,b=5,c=(0.00, 0.75, 0.75))  # 使用默认配置文件
    
    mainflowL(a=3,b=13,c='#EEAD85')
    mainflowL(a=3,b=9,c=(0.00, 0.75, 0.75))
    
    

