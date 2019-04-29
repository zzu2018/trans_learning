# trans_learning
用深度学习实现WiFi信号对动作、身份的识别并可实现在不同环境中适应性的学习

# 项目结构
arguments.py 参数设置
## 原始数据处理
process_data_modify.py  
 
 从input_raw_data_pkg文件夹中读取原始数据，经程序处理后的数据输出到processed_data_pkg文件夹
## 读取数据
read_data_modify.py 

从processed_data_pkg文件夹读取训练及测试模型所需要的数据
## 模型构建
buid_model_stack_conv.py  构建一维卷积模型结构

buid_model_stack_conv2D.py  构建二维卷积模型结构
## 训练模型
train_scratch.py  用一维卷积模型训练数据

train_scratch_conv2D.py  用二维卷积模型训练数据
## 迁移模型 
trans_learning.py  实现数据的迁移训练(二维卷积)