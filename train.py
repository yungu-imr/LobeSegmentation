from model import unet_residual, unet_residual_nopos, unet_residual_fissureAtt
#from modelJIN import unet
from lobe_data import LobeData
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
from utils import generator,PredictCases
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from dice import dice_coef, dice_coef_loss, focal_loss
import time
from utils import predict_cases, MyCbk
import gc



# def predict():
#     data_loader = LobeData(window_size=(72,80,80), moving_size=(40,48,48),
#                              exact09_test_path = './airwayCT/prep_result_noip_EXACTtest/')
#     test_images, test_coors = data_loader.load_exact09_test()
#
#     base_path = './results/u3d201903162249'
#
#     model_path = os.path.join(base_path,'weight.h5')
#     save_path = os.path.join(base_path,'exact09_test')
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     print('load test data  complete>>>>')
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
#         K.set_session(sess)
#         sess.run(tf.global_variables_initializer())
#         model = unet(input_shape = (32,224,224,1))
#         model.load_weights(model_path)
#         predict_cases(model,test_images,[],test_coors,epoch=None,batch_size=6,save_path=save_path,save_only=True)


def train(gpu_num, epochs=15, load_path=None, if_test=False):
    """网络训练函数
    gpu_num：使用的gpu个数(>1)
    epochs:
    load_path:不为None时，将从该路径导入模型参数，在该参数基础再训练
    if_test:为True时，只进行测试。此时load_path不能为None
    """
    batch_size = 3 * gpu_num # 3 * gpu_num
    learing_rate = 1e-3
    print('load data>>>>')
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    # (z,y,x)
    # window_size=(72,80,80), moving_size=(20,24,24),test_moving_size=(40,48,48)  
    data_loader = LobeData(window_size=(32, 224, 224), moving_size=(8, 56, 56), test_moving_size=(16, 112, 112),
                           train_path='./lobe_data2/train/', test_path='./lobe_data2/test/')
    if not if_test:
        train_images, train_labels, train_edges, train_coors = data_loader.load_train_data()
    #train_images, train_labels, train_dists,train_coors, val_images, val_labels, val_dists,val_coors  = data_loader.load_train_data(validation = True)
    test_images, test_labels, test_edges, test_coors, test_names = data_loader.load_test_data()
    train_coor_num = 0
    if not if_test:
        for temp_coors in train_coors:
            train_coor_num += len(temp_coors)
        print ("train_coor_num:", train_coor_num)
    # val_coor_num = 0
    # for temp_coors in val_coors:
    #   val_coor_num += len(temp_coors)
    # print ("val_coor_num:", val_coor_num)
    print('data loading complete!')

    print('model loaded>>>>')
    print('fitting model>>>>')


    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    result_folder=time.strftime("%Y%m%d%H%M", time.localtime())
    save_path=os.path.join('./results', 'unet_res_aug_data2_edge2_FA_GP_fissureAtt_fintune_attpredict' + result_folder)
    log_path = os.path.join(save_path, 'logs')
    mid_path = os.path.join(save_path, 'mid_res')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(mid_path):
        os.mkdir(mid_path)

    tb = TensorBoard(log_path)
    predict_callback = PredictCases(test_images, test_labels, test_coors, test_names, batch_size, save_path=mid_path,
                                    run_epoch=[0, 2, 5, 8, 10, 12, epochs-1])

    #stop = EarlyStopping(patience=4)  
    if if_test:
        model = unet_residual_fissureAtt(input_shape=(32, 224, 224, 1))
        model.load_weights(load_path)
        predict_cases(model, test_images, test_labels, test_coors, test_names, epoch=0, batch_size=batch_size, save_path=save_path)
        # predict_cases(model, test_images, test_labels, test_coors, test_names, epoch=0, batch_size=batch_size, save_path=save_path, save_only=True)
        return
    with tf.device('/cpu:0'):
        model = unet_residual_fissureAtt(input_shape=(32, 224, 224, 1))  #(32, 224, 224, 1)
    if load_path != None:
        model.load_weights(load_path, by_name=True)
    model.summary()
    checkpoint = MyCbk(model, path=save_path)
    if gpu_num == 1:
        with tf.device('/gpu:0'):
            model = unet_residual_fissureAtt(input_shape=(32, 224, 224, 1))
        if load_path != None:
            model.load_weights(load_path, by_name=True)
        checkpoint = MyCbk(model, path=save_path)
        parallel_model = model
    else:
        parallel_model = multi_gpu_model(model, gpus=gpu_num)
    f_loss1 = focal_loss(num_classes = 6)
    f_loss2 = focal_loss(num_classes = 2)
    dice_loss1 = dice_coef_loss([1, 1, 1, 1, 1, 1]) # [0.606, 0.0831, 0.0346, 0.0944, 0.0955, 0.0862]
    dice_loss2 = dice_coef_loss(None, classes_w_t2=[0., 1.])
    loss_weights = [0.7, 0.3] # [0.7, 0.3]  [1.0, 0.0]
    print("save_path:",save_path)
    print("load_path:",load_path)
    print("learing_rate:", learing_rate)
    print("loss_weights:", loss_weights)
    print("dice_loss1:", "dice_loss2")

    parallel_model.compile(optimizer=Adam(lr=learing_rate), loss=[dice_loss1, dice_loss2], loss_weights=loss_weights, metrics=[dice_coef]) #lr=1e-3
    # model.fit(x=image_train,y=label_train,batch_size=2,epochs=20,validation_split=0.1,callbacks=[tb])
    parallel_model.fit_generator(generator=generator(train_images, train_labels, train_edges, train_coors, batch_size=batch_size),
                                 steps_per_epoch=train_coor_num / batch_size,
                                 epochs=epochs,
                                 #validation_data=generator(val_images, val_labels, val_dists,val_coors, batch_size = batch_size),
                                 #validation_steps=val_coor_num / batch_size,
                                 verbose=1,
                                 max_queue_size=20,
                                 # callbacks=[tb, checkpoint, predict_callback]
                                 callbacks=[tb, checkpoint])
    model.save_weights(os.path.join(save_path, 'weight.h5'))
    del train_images, train_labels, train_edges, train_coors, model
    gc.collect()
    model = unet_residual_fissureAtt(input_shape=(32, 224, 224, 1))
    # batch_size = gpu_num * 2
    for ep in [0, 2, 5, 8, 10, 12, epochs-1]:
        model_path = os.path.join(save_path, "model_at_epoch_%d.h5"%ep)
        model.load_weights(model_path)
        predict_cases(model, test_images, test_labels, test_coors, test_names, epoch=ep, batch_size=batch_size, save_path=mid_path)

# def predict(gpu_num):
#     save_path= './results/unet_res_aug_data2_edge2_FA_GP_fissureAtt201912041125'
#     mid_path = os.path.join(save_path, 'mid_res')
#     batch_size = 3 * gpu_num
#     print('load data>>>>')
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(graph=tf.get_default_graph(), config=config)
#     # (z,y,x)
#     # window_size=(72,80,80), moving_size=(20,24,24),test_moving_size=(40,48,48)  
#     data_loader = LobeData(window_size=(32, 224, 224), moving_size=(8, 56, 56), test_moving_size=(16, 112, 112),
#                            train_path='./lobe_data2/train/', test_path='./lobe_data2/test/')
#     test_images, test_labels, test_edges, test_coors, test_names = data_loader.load_test_data()
#     K.set_session(sess)
#     sess.run(tf.global_variables_initializer())
#     model = unet_residual_fissureAtt(input_shape=(32, 224, 224, 1))
#     epochs = 15
#     for ep in [0, 2, 5, 8, 10, 12, epochs-1]:
#         model_path = os.path.join(save_path, "model_at_epoch_%d.h5"%ep)
#         model.load_weights(model_path)
#         predict_cases(model, test_images, test_labels, test_coors, test_names, epoch=ep, batch_size=batch_size, save_path=mid_path)


if __name__ == '__main__':
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    # predict(gpus)
    # exit()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    #load_path = '/media/pami/MyPassport/airwayCT/code/results/u3d_JIN7280201904020829/model_at_epoch_2.h5'
    # load_path = './results/unet_residual_201908241131/model_at_epoch_14.h5'
    # load_path = './results/unet_res_aug_edge2_dice201909272249/model_at_epoch_4.h5'
    # load_path = './results/unet_res_aug_edge2_dice201909301333/model_at_epoch_10.h5'
    # load_path = './results/unet_res_aug_edge2_data2201910041315/model_at_epoch_8.h5'
    # load_path = './results/unet_res_aug_data2_edge2_FA_GP_fissureAtt201912041125/model_at_epoch_14.h5'
    load_path = './results/unet_res_aug_data2_edge2_FA_GP_fissureAtt_fintune201912061746/model_at_epoch_0.h5'
    # load_path = None
    train(gpus, load_path=load_path, if_test=True)
    #predict()