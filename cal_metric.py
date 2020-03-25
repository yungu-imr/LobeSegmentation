import numpy as np
import SimpleITK as sitk
import os
from dice import dice_coef_np


def metric(y_true, y_pred):
    eposion = 1e-8
    num_classes = 6
    soft_dice, soft_lobe_dices = dice_coef_np(y_true, y_pred, threshold_mode=False)
    hard_dice, hard_lobe_dices = dice_coef_np(y_true, y_pred, threshold_mode=True)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    res = dict()
    res['hard_dice'] = hard_dice
    res['hard_lobe_dices'] = hard_lobe_dices
    res['soft_lobe_dices'] = soft_lobe_dices
    res['soft_dice'] = soft_dice
    res['sen'] = 0.0
    res['fpr'] = 0.0
    res['ppv'] = 0.0
    for i in range(1, num_classes):
        y_true_f = (y_true == i)
        y_pred_f = (y_pred == i)
        tp = np.sum(y_true_f * y_pred_f)
        fp = np.sum(y_pred_f) - tp
        res['sen'] += tp / float(np.sum(y_true_f))
        res['fpr'] += fp / float(fp + np.size(y_true_f) - np.sum(y_true_f))
        res['ppv'] += tp / (float(np.sum(y_pred_f))+eposion)
    res['sen'] /= (num_classes - 1)
    res['fpr'] /= (num_classes - 1)
    res['ppv'] /= (num_classes - 1)
    return res


if __name__ == '__main__':
    ### private data result
    result_paths = ['/media/pami4/MyPassport/airwayCT/code_conn/results/u3d_dist201903190854/mid_res/',
                    '/media/pami4/MyPassport/airwayCT/code/results/u3d_dist201903181245/mid_res/'] 
    exp_names = ['unet_dist','unet_raw']
    #result_paths = ['/media/pami4/MyPassport/airwayCT/code_conn/results/u3d_dist201903192144/mid_res/']
    #exp_names = ['unet_connectivity']
    gt_path = '/media/pami4/MyPassport/airwayCT/code/airwayCT/self_prep_all/'
    gts = []
    for i in range(1, 11):
        itk_img = sitk.ReadImage(os.path.join(gt_path, "case%d_label.nii"%i))
        img = sitk.GetArrayFromImage(itk_img)
        gts.append(img) 
    file = open(os.path.join('.', 'cal_metric.txt'),'a')
    for result_path,exp_name in zip(result_paths,exp_names):
        result_list = os.listdir(result_path)
        file.write(exp_name + ' --> ' + result_path + '\n')
        print(exp_name + ' --> ' + result_path)
        result_list = [name for name in result_list if name.startswith('e010')]#e014
        result_list.sort()
        soft_dices = []
        hard_dices= []
        sens = []
        fprs = []
        ppvs = []
        ##
        first_element = result_list.pop(0)
        result_list.append(first_element)
        ##
        for idx,case_name in enumerate(result_list):
            predict_img = sitk.ReadImage(os.path.join(result_path, case_name))
            predict_img = sitk.GetArrayFromImage(predict_img)
            res = metric(gts[idx], predict_img)
            soft_dices.append(res['soft_dice'])
            hard_dices.append(res['hard_dice'])
            sens.append(res['sen'])
            fprs.append(res['fpr'])
            ppvs.append(res['ppv'])
            res_str = "\tcase:%02d, hard_dice:%.4f, soft_dice:%.4f, sen:%.4f, ppv:%.4f,fpr:%.4f"%(idx+1,res['soft_dice'],res['hard_dice'],res['sen'],res['ppv'],res['fpr'])
            file.write(res_str + '\n')
            print(res_str)
        m_hard_dice = sum(hard_dices) / float(len(hard_dices))
        m_soft_dice = sum(soft_dices) / float(len(soft_dices))
        m_sen = sum(sens) / float(len(sens))
        m_fpr = sum(fprs) / float(len(fprs))
        m_ppv = sum(ppvs) / float(len(ppvs))
        std_hard_dice = np.std(np.array(hard_dices))
        std_soft_dice = np.std(np.array(soft_dices))
        std_sen = np.std(np.array(sens))
        std_fpr = np.std(np.array(fprs))
        std_ppv = np.std(np.array(ppvs))
        res_str = "\tmean hard_dice:%.4f, mean soft_dice:%.4f, mean sen:%.4f, mean ppv:%.4f, mean fpr:%.4f"%(m_hard_dice,m_soft_dice,m_sen,m_ppv,m_fpr)
        print (res_str)
        res_str2 = "\tstd hard_dice:%.4f, std soft_dice:%.4f, std sen:%.4f, std ppv:%.4f, std fpr:%.4f"%(std_hard_dice,std_soft_dice,std_sen,std_ppv,std_fpr)
        print(res_str2)
        file.write(res_str + '\n\n')
        file.write(res_str2 + '\n\n')



# def predict_cases(model,images,masks,dists,coors,batch_size,epoch,save_path,save_only = False,lung_masks=None):
#   soft_dices = []
#   hard_dices= []
#   sens = []
#   fprs = []
#   if not save_only:
#       file = open(os.path.join(save_path, 'result.txt'),'a')
#   for i in range(len(images)):
#       case_image = images[i]
#       if not save_only:
#           case_mask = masks[i] 
#       case_coors = coors[i]
#       case_dists = dists[i]
#       coor_num = len(case_coors)
#       steps = (coor_num + batch_size - 1) // batch_size
#       res = model.predict_generator(generator_for_test([case_image],[case_dists],[case_coors],batch_size),
#                                    steps= steps,
#                                    max_queue_size=20,
#                                    verbose=0)
#       #print ("predict_generator done")
#       res = predict_conncube(res)
#       predict_count = np.zeros_like(case_image,dtype=np.float32)
#       predict_res = np.zeros_like(case_image,dtype=np.float32)
#       for idx,coor in enumerate(case_coors):
#           predict_res[coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1] += res[idx]
#           predict_count[coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1] += 1.0

#       predict_res = predict_res / predict_count
#       if not save_only:
#           soft_dice = dice_coef_np(case_mask, predict_res)
#           predict_res = np.where(predict_res > 0.5, 1.0, 0.0)
#           hard_dice = dice_coef_np(case_mask, predict_res)
#           soft_dices.append(soft_dice)
#           hard_dices.append(hard_dice)
#           me = metric(case_mask, predict_res)
#           sens.append(me['sen'])
#           fprs.append(me['fpr'])
#           save_itk(predict_res[...,0],os.path.join(save_path, "e%03dcase%02d_h%.4f_s%.4f.nii.gz"%(epoch,i+1,hard_dice,soft_dice)))
#           res_str = "epoch:%03d, case:%02d, hard_dice:%.4f, soft_dice:%.4f, sen:%.4f, fpr:%.4f"%(epoch,i+1,hard_dice,soft_dice,me['sen'],me['fpr'])
#           print (res_str)
#           file.write(res_str + '\n')
#       else:
#           if lung_masks != None:
#               predict_res = predict_res * lung_masks[i]
#           predict_res = np.where(predict_res > 0.5, 1.0, 0.0)
#           save_itk(predict_res[...,0],os.path.join(save_path, "CASE%02d_predict.nii.gz"%(i+1)))
#           print ("case:%02d done"%(i+1+20))
#   if not save_only:
#       m_hard_dice = sum(hard_dices) / float(len(hard_dices))
#       m_soft_dice = sum(soft_dices) / float(len(soft_dices))
#       m_sen = sum(sens) / float(len(sens))
#       m_fpr = sum(fprs) / float(len(fprs))
#       res_str = "epoch:%03d,mean hard_dice:%.4f, mean soft_dice:%.4f, mean sen:%.4f, mean fpr:%.4f"%(epoch,m_hard_dice,m_soft_dice,m_sen,m_fpr)
#       print (res_str)
#       file.write(res_str + '\n\n')
#       file.close()