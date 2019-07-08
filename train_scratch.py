import time
import keras

from buid_model_GRU import build_model
from read_data_modify import read_data, return_labels
from utils.utils import *
from arguments import *

# 添加GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

for data_name in ALL_DATA_NAMES:
    print('正在训练-- ', data_name, ' --数据集')
    train_data_dir, activity_list = return_labels(data_name)  # 返回data_name数据集所在的文件目录和数据集的标签
    print('activity_list: ', activity_list)
    x_train, y_train, x_test, y_test = read_data(train_data_dir, activity_list)
    print('x_train.shape', x_train.shape)
    # save orignal y because later we will use binary
    y_true_val = None
    y_pred_val = None
    y_true = np.argmax(y_test, 1)

    if len(x_train.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    start_time = time.time()

    input_shape = (None, x_train.shape[2])
    print('input_shape', input_shape)
    nb_classes = len(activity_list)
    print('nb_classes: ', nb_classes)

    model = build_model(input_shape=input_shape, nb_classes=nb_classes, pre_model_path=None,
                        freezen=False, freezen_layers=None)

    # reduce learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)
    # model checkpoint
    ckp_metric_file_path = result_save_path + data_name + '/'
    ckp_metric_file_path = create_directory(ckp_metric_file_path)
    print('模型输出文件夹：', ckp_metric_file_path)

    save_model_path = ckp_metric_file_path + 'model-io-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-' \
                                             'val_acc{val_acc:.5f}.h5f'

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_acc',
                                                       save_best_only=False, verbose=1, mode='max', period=10)
    tensorboard = keras.callbacks.TensorBoard(log_dir=ckp_metric_file_path + '/log/')
    callbacks = [reduce_lr, model_checkpoint, tensorboard]

    if verbose is True:
        model.summary()

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, shuffle=True,
                     verbose=verbose, validation_data=(x_test, y_test), callbacks=callbacks)

    best_val_model_path = choose_best_model(result_save_path + data_name)
    model = keras.models.load_model(best_val_model_path)

    y_pred = model.predict(x_test)
    # convert the predicted from binary to integer
    y_pred = np.argmax(y_pred, axis=1)

    duration = time.time() - start_time
    df_metrics = save_logs(ckp_metric_file_path, hist, y_pred, y_true, duration, y_true_val, y_pred_val)

    print(df_metrics)

keras.backend.clear_session()

# filepath = h5fy_path + 'model-io-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max', period=20)
