import keras

from read_data_modify import read_data, return_labels
from utils.utils import *
from arguments import *

# 添加GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

models_path = result_save_path + '*'
models_dir = sorted(glob.glob(models_path))
models_dirName = []
for dir in models_dir:
    name = dir.split('\\')[-1]
    models_dirName.append(name)
print('models_dirName: ', models_dirName)
# exit()
for data_name in models_dirName:
    best_val_model_path = choose_best_model(result_save_path+data_name)

    for trans_data_name in ALL_TEST_DATA_NAMES:
        # if data_name == trans_data_name:
        #     continue
        print('正在用-' + trans_data_name + '-数据集测试-' + data_name + '-训练好的模型')
        train_data_dir, activity_list = return_labels(trans_data_name)  # 返回data_name数据集所在的文件目录和数据集的标签
        _, _, x_test, y_test = read_data(train_data_dir, activity_list)

        y_true = np.argmax(y_test, 1)

        model = keras.models.load_model(best_val_model_path)

        y_pred = model.predict(x_test)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        model_predict_logs_dir = 'model_predict_logs/'+trans_data_name+'_predict_on_'+data_name+'/'
        model_predict_logs_dir = create_directory(model_predict_logs_dir)
        print('模型输出文件夹：', model_predict_logs_dir)
        df_metrics = compute_metrics(model_predict_logs_dir, y_pred, y_true)

        print(df_metrics)

keras.backend.clear_session()
