batch_size = 16# 批数据大小
nb_epochs = 600 # 模型训练迭代次数
verbose = True


threshold = 60  # 含有有效动作的阈值大小

is_data_halve = False  # 是否将处理后的数据再次减半


slide_win_size = 600  # 滑动窗口大小

slide_steps = 200   # 每次滑动的步数

carrier_nums = 90  # 子载波总数

input_shape = (int(carrier_nums/3), slide_win_size, 3)
kk = 10  # 测试集的比例 10/100

ALL_DATA_NAMES = ['WiFi_data_old_50']  # 所有数据集的名称
ALL_TEST_DATA_NAMES = ['vData2', 'vData3']  # 所有数据集的名称


result_save_path = 'checkpoint/test_scratch/new_idea_1/'  # 数据训练后， 模型保存路径文件夹

trans_learning_save_path = 'checkpoint/trans_learning/'  # 数据迁移学习后，模型保存路径文件夹

INPUT_RAW_DATA_PKG = 'input_raw_data_pkg/'  # raw data directory

INPUT_PROCESSED_DATA_PKG = 'processed_data_pkg/'  # processed data directory
