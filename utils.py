import numpy as np
from keras import callbacks
import math
#from dice import dice_coef_np
import SimpleITK as sitk
import os
from scipy.ndimage import gaussian_filter, rotate, zoom
# from skimage import transform
import random
from cal_metric import metric


def augment(sample, label, edge, fissure_prob):
	"""数据增强
    sample：CT图像
    label：肺叶标记
    edge：肺裂隙边界
    fissure_prob：肺裂隙概率图
    """
    ## z, y, x, 1
    flipid = np.random.randint(2)*2-1
    if flipid == -1:
        sample = np.ascontiguousarray(sample[:,:,::flipid,:])
        label = np.ascontiguousarray(label[:,:,::flipid,:])
        edge = np.ascontiguousarray(edge[:,:,::flipid,:])
        # dist = np.ascontiguousarray(dist[:,:,::flipid,:])
        fissure_prob = np.ascontiguousarray(fissure_prob[:,:,::flipid,:])
    # else:
    #     if random.random() > 0.7:
    #         degree = np.random.randint(21) - 10
    #         sample = rotate(sample, degree, axes=(1, 2), reshape=False)
    #         label = rotate(label, degree, axes=(1, 2), reshape=False)
    #         edge = rotate(edge, degree, axes=(1, 2), reshape=False)
    return sample, label, edge, fissure_prob  #, dist
# def augment(sample, label, edge, dist, fissure_prob):
#     ## z, y, x, 1
#     flipid = np.random.randint(2)*2-1
#     if flipid == -1:
#         sample = np.ascontiguousarray(sample[:,:,::flipid,:])
#         label = np.ascontiguousarray(label[:,:,::flipid,:])
#         edge = np.ascontiguousarray(edge[:,:,::flipid,:])
#         dist = np.ascontiguousarray(dist[:,:,::flipid,:])
#         fissure_prob = np.ascontiguousarray(fissure_prob[:,:,::flipid,:])
#     # else:
#     #     if random.random() > 0.7:
#     #         degree = np.random.randint(21) - 10
#     #         sample = rotate(sample, degree, axes=(1, 2), reshape=False)
#     #         label = rotate(label, degree, axes=(1, 2), reshape=False)
#     #         edge = rotate(edge, degree, axes=(1, 2), reshape=False)
#     return sample, label, edge, dist, fissure_prob  #, dist


# masks, images and coors is list. len(images) = len(coors)
def generator(images, masks, edges, coors, batch_size, shuffle=True):
	"""训练过程的数据生成器
    images：（CT图像，肺裂隙概率图）
    masks：肺叶标记
    edges：肺裂隙边界
    coors：cropped_cube的坐标的list
    batch_size:
    shuffle:训练过程中是否shuffle
    """
    images, fissure_probs = images
    coor_to_case = []
    all_coors = []
    num_classes = 6
    for idx, case_coor in enumerate(coors):
        case_coor_num = len(case_coor)
        coor_to_case = coor_to_case + [idx] * case_coor_num
        all_coors = all_coors + case_coor
    # x_batch, y_batch = [], []
    x_patch_size = [coors[0][0][3]-coors[0][0][0], coors[0][0][4] - coors[0][0][1], coors[0][0][5] - coors[0][0][2], 1]
    y_patch_size = [coors[0][0][3]-coors[0][0][0], coors[0][0][4] - coors[0][0][1], coors[0][0][5] - coors[0][0][2], num_classes]
    edge_patch_size = [coors[0][0][3]-coors[0][0][0], coors[0][0][4] - coors[0][0][1], coors[0][0][5] - coors[0][0][2], 2]
    x_batch = np.zeros([batch_size]+x_patch_size, dtype=np.float32)
    y_batch = np.zeros([batch_size]+y_patch_size, dtype=np.float32)
    edges_batch = np.zeros([batch_size]+edge_patch_size, dtype=np.float32)
    # dist_batch = np.zeros([batch_size,patch_size[0],patch_size[1],patch_size[2],4],dtype= np.float32)
    dist_batch = np.zeros([batch_size,x_patch_size[0],x_patch_size[1],x_patch_size[2],3],dtype= np.float32)
    fissure_prob_batch = np.zeros([batch_size]+x_patch_size, dtype=np.float32)
    count = 0
    while True:
        index = np.arange(len(all_coors))
        if shuffle:
            np.random.shuffle(index)

        for i in index:
            if count < batch_size:
                coor = all_coors[i]
                # img = images[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1]
                # msk = masks[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1]
                # x_batch.append(img)
                # y_batch.append(msk)
                curx = images[coor_to_case[i]][coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :]
                cury = masks[coor_to_case[i]][coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :]
                curedge = edges[coor_to_case[i]][coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :]
                curfissure_prob = fissure_probs[coor_to_case[i]][coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :]
                prob_aug = random.random()
                if prob_aug > 0.8:
                    '''
                    DATA AUGMENTATION GAUSSION
                    '''
                    curx = gaussian_filter(curx, sigma=1)

                start = [coor[0], coor[1], coor[2]]
                normstart = np.array(start).astype('float32')/np.array(images[coor_to_case[i]].shape[0:3])-0.5
                crop_size = [x_patch_size[0], x_patch_size[1], x_patch_size[2]]
                crop_size = np.array(crop_size) # downsample
                normsize = np.array(crop_size).astype('float32')/np.array(images[coor_to_case[i]].shape[0:3])
                xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(crop_size[0])),
                                       np.linspace(normstart[1],normstart[1]+normsize[1],int(crop_size[1])),
                                       np.linspace(normstart[2],normstart[2]+normsize[2],int(crop_size[2])),indexing ='ij')
                coords = np.concatenate([xx[...,np.newaxis], yy[...,np.newaxis],zz[...,np.newaxis]],-1).astype('float32')
                # dist = dists[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1]
                # dist = np.concatenate([coords,dist],axis = -1)
                dist = coords

                curx, cury, curedge, curfissure_prob= augment(curx, cury, curedge, curfissure_prob)
                # curx, cury, curedge, dist, curfissure_prob= augment(curx, cury, curedge, dist, curfissure_prob)
                x_batch[count, ...] = curx
                y_batch[count, ...] = cury
                dist_batch[count,...] = dist
                edges_batch[count,...] = curedge
                fissure_prob_batch[count,...] = curfissure_prob
                #print(dist_batch.shape)
                count += 1
            if count == batch_size:
                yield [x_batch, dist_batch, fissure_prob_batch], [y_batch, edges_batch]
                #yield x_batch, y_batch
                count = 0
                # x_batch, y_batch = [], []
                x_batch = np.zeros([batch_size]+x_patch_size, dtype=np.float32)
                y_batch = np.zeros([batch_size]+y_patch_size, dtype=np.float32)
                dist_batch = np.zeros([batch_size,x_patch_size[0],x_patch_size[1],x_patch_size[2],3],dtype= np.float32)
                edges_batch = np.zeros([batch_size]+edge_patch_size, dtype=np.float32)
                fissure_prob_batch = np.zeros([batch_size]+x_patch_size, dtype=np.float32)


def generator_for_test(images, fissure_probs, coors, batch_size):
	"""测试过程的数据生成器
    images：CT图像
    fissure_probs：肺裂隙概率图
    coors：cropped_cube的坐标的list
    batch_size:
    """
    coor_to_case = []
    all_coors = []
    num_classes = 6
    for idx, case_coor in enumerate(coors):
        case_coor_num = len(case_coor)
        coor_to_case = coor_to_case + [idx] * case_coor_num
        all_coors = all_coors + case_coor
    # x_batch = []
    x_patch_size = [coors[0][0][3]-coors[0][0][0], coors[0][0][4] - coors[0][0][1], coors[0][0][5] - coors[0][0][2], 1]
    x_batch = np.zeros([batch_size]+x_patch_size,dtype= np.float32)
    dist_batch = np.zeros([batch_size,x_patch_size[0],x_patch_size[1],x_patch_size[2],3],dtype= np.float32)
    fissure_prob_batch = np.zeros([batch_size]+x_patch_size, dtype=np.float32)
    count = 0
    coor_num = len(all_coors)
    index = np.arange(coor_num)
    for i in index:
        if count < batch_size:
            coor = all_coors[i]
            # img = images[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1]
            # x_batch.append(img)
            x_batch[count,...] = images[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5], :]
            fissure_prob_batch[count,...] = fissure_probs[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5], :]
            start = [coor[0], coor[1], coor[2]]
            normstart = np.array(start).astype('float32')/np.array(images[coor_to_case[i]].shape[0:3])-0.5
            crop_size = [x_patch_size[0], x_patch_size[1], x_patch_size[2]]
            crop_size = np.array(crop_size) # downsample
            normsize = np.array(crop_size).astype('float32')/np.array(images[coor_to_case[i]].shape[0:3])
            xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(crop_size[0])),
                                   np.linspace(normstart[1],normstart[1]+normsize[1],int(crop_size[1])),
                                   np.linspace(normstart[2],normstart[2]+normsize[2],int(crop_size[2])),indexing ='ij')
            coords = np.concatenate([xx[...,np.newaxis], yy[...,np.newaxis],zz[...,np.newaxis]],-1).astype('float32')
            # dist = dists[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1]
            # dist = np.concatenate([coords,dist],axis = -1)
            dist = coords
            dist_batch[count,...] = dist
            #print(dist_batch.shape)            
            count += 1
        if count == batch_size or i == coor_num - 1:
            if i == coor_num - 1:
                while count < batch_size:
                    # x_batch.append(img)
                    x_batch[count,...] = images[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5], :]
                    fissure_prob_batch[count,...] = fissure_probs[coor_to_case[i]][coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5], :]
                    dist_batch[count,...] = dist
                    count += 1
            yield [x_batch, dist_batch, fissure_prob_batch]
            #yield x_batch
            count = 0
            # x_batch = []
            x_batch = np.zeros([batch_size]+x_patch_size,dtype= np.float32)
            dist_batch = np.zeros([batch_size,x_patch_size[0],x_patch_size[1],x_patch_size[2],3],dtype= np.float32)
            fissure_prob_batch = np.zeros([batch_size]+x_patch_size, dtype=np.float32)


def predict_cases(model, images, masks, coors, case_names, batch_size, epoch, save_path, lung_masks=None, save_only=False):
	"""预测新的CT图像的肺叶，
    model：网络模型
    images：（CT图像，肺裂隙概率图）
    masks：肺叶标记
    coors：cropped_cube的坐标的list
    case_names：CT图像名字，与images对应
    batch_size:
    epoch：保存预测结果时，文件的名字的一部分
    save_path：预测结果保存的路径
    lung_masks：肺实质区域的mask。不为None时，会根据这个mask去掉不在肺实质内的预测结果。
	当save_only=Flase时，masks不能为None，该函数预测肺叶并保存结果，同时计算预测结果的评价指标
	当save_only=True时,该函数只预测肺叶，并保存结果
	"""
    images, fissure_probs = images
    soft_dices = []
    hard_dices = []
    soft_lobe_dices = []
    hard_lobe_dices = []
    sensitivities = []
    ppvs = []
    num_classes = 6
    for i in range(len(images)):
        case_image = images[i]
        case_fissure_probs = fissure_probs[i]
        case_name = case_names[i]
        if not save_only:
            case_mask = masks[i]
        case_coors = coors[i]
        # case_dists = dists[i]
        coor_num = len(case_coors)
        steps = (coor_num + batch_size - 1) // batch_size
        res, edges = model.predict_generator(generator_for_test([case_image],[case_fissure_probs],[case_coors],batch_size),
                                      steps=steps,
                                      max_queue_size=20,
                                      verbose=0)
        # res = sig_att
        res_shape = list(case_image.shape)
        res_shape[-1] = num_classes
        predict_count = np.zeros(res_shape, dtype=np.float32)
        predict_res = np.zeros(res_shape, dtype=np.float32)
        for idx, coor in enumerate(case_coors):
            # res_resized = zoom(res[idx],(2,2,2,1), order=1)
            # predict_res[coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :] += res_resized
            predict_res[coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :] += res[idx]
            predict_count[coor[0]:coor[3], coor[1]:coor[4], coor[2]:coor[5], :] += 1.0

        predict_res = predict_res / predict_count
        if not save_only:
            me = metric(case_mask, predict_res)
            #soft_dice = dice_coef_np(case_mask, predict_res)
            predict_res = np.argmax(predict_res, axis=-1)
            predict_res = predict_res.astype(np.int32)
            #hard_dice = dice_coef_np(case_mask, predict_res)
            soft_dices.append(me['soft_dice'])
            hard_dices.append(me['hard_dice'])
            soft_lobe_dices.append(me['soft_lobe_dices'])
            hard_lobe_dices.append(me['hard_lobe_dices'])
            sensitivities.append(me['sen'])
            ppvs.append(me['ppv'])
            save_itk(predict_res, os.path.join(save_path, "e%03d%s_h%.4f_s%.4f_se%.4f_ppv%.4f.nii.gz"%(epoch,case_name,me['hard_dice'],me['soft_dice'],me['sen'],me['ppv'])))
            print ("%s, hard_dice:%.4f,soft_dice:%.4f,sensitivity:%.4f,ppv:%.4f"%(case_name, me['hard_dice'],me['soft_dice'],me['sen'],me['ppv']))
        else:
            if lung_masks!= None:
                predict_res = predict_res * lung_masks[i]
            predict_res = np.argmax(predict_res, axis=-1)
            predict_res = predict_res.astype(np.int32)
            # predict_res = predict_res[:,:,:,0]
            save_itk(predict_res, os.path.join(save_path, "%s_predict.nii.gz" % case_name))
            print ("case:%s done"%case_name)
    if not save_only:
        soft_lobe_dices = np.array(soft_lobe_dices)
        hard_lobe_dices = np.array(hard_lobe_dices)
        m_soft_lobe_dices = np.mean(soft_lobe_dices, axis=0)
        m_hard_lobe_dices = np.mean(hard_lobe_dices, axis=0)
        m_hard_dice = sum(hard_dices) / float(len(hard_dices))
        m_soft_dice = sum(soft_dices) / float(len(soft_dices))
        m_sen = sum(sensitivities) / float(len(sensitivities))
        m_ppv = sum(ppvs) / float(len(sensitivities))
        print ("per lobe hard dice", m_hard_lobe_dices, "per lobe soft dice", m_soft_lobe_dices)
        print ("mean hard_dice:%.4f, mean soft_dice:%.4f, mean sensitivity:%.4f, mean ppv:%.4f"%(m_hard_dice,m_soft_dice,m_sen,m_ppv))
        with open(os.path.join(save_path, 'result.txt'), 'a') as file:
            file.write("epoch:%03d, mean hard_dice:%.4f, mean soft_dice:%.4f,  mean sensitivity:%.4f, mean ppv:%.4f \n" % (epoch, m_hard_dice, m_soft_dice, m_sen, m_ppv))
            file.write("epoch:%03d, per lobe hard dice:%s, per lobe soft dice:%s"%(epoch, np.array_str(m_hard_lobe_dices), np.array_str(m_soft_lobe_dices)))

def save_itk(images, save_path):
    out = sitk.GetImageFromArray(images)
    sitk.WriteImage(out, save_path)


class PredictCases(callbacks.Callback):
    """测试时的回调类"""
    def __init__(self, images, masks, coors, test_names, batch_size, save_path, run_epoch=None):
        """
        run_epoch:一个list，当训练到该list内的epoch时才进行测试
        """
        super(PredictCases, self).__init__()
        self.run_epoch = run_epoch
        self.images = images
        self.masks = masks
        self.coors = coors
        self.test_names = test_names
        self.batch_size = batch_size
        self.save_path = save_path
        # self.dists = dists

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.run_epoch:
            predict_cases(self.model, self.images, self.masks, self.coors, self.test_names, self.batch_size, epoch,
                          self.save_path)


class MyCbk(callbacks.Callback):
    """保存模型参数的回调类"""
    def __init__(self, model, path):
        super(MyCbk, self).__init__()
        self.model_to_save = model
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save_weights(os.path.join(self.path, 'model_at_epoch_%d.h5' % epoch))