import matplotlib.pyplot as plt
import numpy as np
import re

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取日志文件
with open('/home/qwe/data/wenxiaoyuan/UA-MT-master/model/UAMT_unlabel/log.txt', 'r') as file:
    log_lines = file.readlines()

# 初始化存储数据的列表
iterations = []
total_losses = []
seg_losses = []
edge_losses = []
cons_dists = []
loss_weights = []

# 日志行的正则表达式模式
log_pattern = re.compile(
    r'iteration (\d+) : total_loss : ([\d\.]+) seg_loss : ([\d\.]+) edge_loss : ([\d\.]+) cons_dist: ([\d\.]+), loss_weight: ([\d\.]+)'
)

# 解析日志文件内容
for line in log_lines:
    match = log_pattern.search(line)
    if match:
        try:
            iter_num = int(match.group(1))
            total_loss = float(match.group(2))
            seg_loss = float(match.group(3))
            edge_loss = float(match.group(4))
            cons_dist = float(match.group(5))
            loss_weight = float(match.group(6))

            # 打印解析结果以检查数据
            print(f"Iteration: {iter_num}, Total Loss: {total_loss}, Seg Loss: {seg_loss}, Edge Loss: {edge_loss}, Cons Dist: {cons_dist}, Loss Weight: {loss_weight}")

            iterations.append(iter_num)
            total_losses.append(total_loss)
            seg_losses.append(seg_loss)
            edge_losses.append(edge_loss)
            cons_dists.append(cons_dist)
            loss_weights.append(loss_weight)
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(f"Exception: {e}")
            continue

# 检查解析后的数据
print("Iterations:", iterations)
print("Total Losses:", total_losses)
print("Seg Losses:", seg_losses)
print("Edge Losses:", edge_losses)
print("Cons Dists:", cons_dists)
print("Loss Weights:", loss_weights)

# 如果数据列表为空，提示用户
if not iterations:
    print("No valid data found in the log file.")
else:
    # 绘制损失随迭代次数的变化图
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(iterations, total_losses, label='total_loss')
    plt.xlabel('iter')
    plt.ylabel('total_loss')
    plt.title('total_loss_with_iter')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(iterations, seg_losses, label='seg_loss', color='orange')
    plt.xlabel('iter')
    plt.ylabel('seg_loss')
    plt.title('seg_loss_with_iter')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(iterations, edge_losses, label='edge_loss', color='green')
    plt.xlabel('iter')
    plt.ylabel('edge_loss')
    plt.title('edge_loss_with_iter')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(iterations, cons_dists, label='con_dist', color='red')
    plt.xlabel('iter')
    plt.ylabel('con_dist')
    plt.title('con_dist_with_iter')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_plot.png')  # 保存图像为文件
    plt.show()

    # 计算edge_loss的基本统计数据以了解其变异性
    edge_loss_mean = np.mean(edge_losses)
    edge_loss_std = np.std(edge_losses)

    print("边缘损失的平均值：", edge_loss_mean)
    print("边缘损失的标准差：", edge_loss_std)
