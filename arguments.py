batch_size = 16
nb_epochs = 150
verbose = True

# 含有有效动作的阈值大小
threshold = 60
# 是否将处理后的数据再次减半
is_data_halve = True

# 滑动窗口大小
slide_win_size = 1000
# 每次滑动的步数
slide_steps = 500  # less than window_size!!!
# 子载波总数
carrier_nums = 90

kk = 30  # 测试集的比例 10/100

# all data names
ALL_DATA_NAMES = ['vData2', 'vData3']  # 所有数据集的名称
ALL_TEST_DATA_NAMES = ['vData2', 'vData3']  # 所有数据集的名称

result_save_path = 'checkpoint/train_scratch/'
trans_learning_save_path = 'checkpoint/trans_learning/'
# raw data directory
INPUT_RAW_DATA_PKG = 'input_raw_data_pkg/'
# processed data directory
INPUT_PROCESSED_DATA_PKG = 'processed_data_pkg/'
