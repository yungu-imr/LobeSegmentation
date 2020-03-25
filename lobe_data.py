import SimpleITK as sitk
import numpy as np
import os
from keras.utils import to_categorical

def sliding_window(data_shape, window_size, moving_size):
	"""滑窗的形式产生cropped cube坐标
	data_shape：CT的大小
	window_size：cropped cube的大小
	moving_size：滑动大小
	"""
    coor = []
    for i in range(3):
        res_sizei = (data_shape[i] - window_size[i]) // moving_size[i] + 1
        coori = [k*moving_size[i] for k in range(res_sizei)]
        if len(coori) == 0:
            print (data_shape)
        if coori[-1] != data_shape[i] - window_size[i]:
            coori.append(data_shape[i] - window_size[i])
        coor.append(coori)
    left_right_top = []
    for c0 in coor[0]:
        for c1 in coor[1]:
            for c2 in coor[2]:
                left_right_top.append((c0, c1, c2, c0+window_size[0], c1+window_size[1], c2+window_size[2]))

    return left_right_top


class LobeData(object):

    def __init__(self, window_size, moving_size, test_moving_size, train_path='./lobe_data/train/',
                 test_path='./lobe_data/test/'):
	    """
		window_size：cropped cube的大小
		moving_size：训练时的滑动大小
		test_moving_size：测试时的滑动大小
		train_path：训练数据所在路径
		test_path：测试数据所在路径
		"""
        super(LobeData, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.window_size = window_size
        self.moving_size = moving_size
        self.test_moving_size = test_moving_size
        self.num_classes = 6

    def one_hot(self, label, num_classes):
    	"""将label改成one hot形式
		num_classes：类别数
    	"""
        new_shape = label.shape + (num_classes,)
        label = label.flatten()
        label = to_categorical(label, num_classes)
        one_hot_label = np.reshape(label, new_shape)
        return one_hot_label

    def load_train_data(self, validation=False):
        file_names = os.listdir(self.train_path)
        label_file_names = [f for f in file_names if f.endswith('_label.nii.gz')]
        label_file_names.sort()
        label_file_names = label_file_names
        data_file_names = [f.replace('_label', '') for f in label_file_names]
        file_num = len(data_file_names)

        dataset_coors = []  # shape = (dataset_size, VOI_nums, 3)
        images = []
        labels = []
        edges = []
        fissure_pros = []
        #
        positive_num = 0.0
        voxel_num = 0.0
        class_num = np.zeros((6,))
        #
        for data_name, label_name in zip(data_file_names, label_file_names):
            print (data_name)
            # if data_name.endswith("7742.nii.gz"):
            #     print (data_name, "pass")
            #     continue
            itk_img = sitk.ReadImage(os.path.join(self.train_path, data_name))
            img = sitk.GetArrayFromImage(itk_img)
            itk_label = sitk.ReadImage(os.path.join(self.train_path, label_name))
            label = sitk.GetArrayFromImage(itk_label)
            edge_name = label_name.replace('_label', '_edge2')
            itk_edge = sitk.ReadImage(os.path.join(self.train_path, edge_name))
            edge = sitk.GetArrayFromImage(itk_edge)
            fissure_pro_name = label_name.replace('_label', '_fissure_prob')
            itk_fissure_pro = sitk.ReadImage(os.path.join(self.train_path, fissure_pro_name))
            fissure_pro = sitk.GetArrayFromImage(itk_fissure_pro)
            edge = self.one_hot(edge, 2)
            positive_num += np.sum(label > 0)
            label = self.one_hot(label, self.num_classes)
            img = img.astype(np.float32)
            img = img.astype(np.float32) / 255.0
            label = label.astype(np.float32)
            # label_num = np.sum(label, axis=(0,1,2))
            # class_num = class_num + label_num
            edge = edge.astype(np.float32)
            fissure_pro = fissure_pro.astype(np.float32)
            voxel_num += img.size
            
            # label = np.pad(label, 1, 'constant', constant_values=0)
            img = img[..., np.newaxis]
            fissure_pro = fissure_pro[..., np.newaxis]
            # label = label[..., np.newaxis]
            images.append(img)
            labels.append(label)
            edges.append(edge)
            fissure_pros.append(fissure_pro)
            coors = sliding_window(img.shape, self.window_size, self.moving_size)
            num = len(coors)
            filtered_coors = []
            for i in range(num):
                coor = coors[i]
                temp_label = label[coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5],0:1]
                if np.sum(temp_label > 0) > 100:# and np.sum(temp_label) < 5000:###150
                    filtered_coors.append(coor)
            dataset_coors.append(filtered_coors)
        # print("class_num", class_num)
        # class_num = class_num / np.sum(class_num)
        # print("class_num", class_num)
        print("positive rate:%f" % (positive_num/voxel_num))
        if validation:
            return (images[0:file_num-1],fissure_pros[0:file_num-1]), labels[0:file_num-1], edges[0:file_num-1],\
                   dataset_coors[0:file_num-1], (images[file_num-1:file_num], fissure_pros[file_num-1:file_num]), labels[file_num-1:file_num], edges[file_num-1:file_num],\
                   dataset_coors[file_num-1:file_num]
        else:
            return (images,fissure_pros), labels, edges, dataset_coors

    def load_test_data(self):
        file_names = os.listdir(self.test_path)
        label_file_names = [f for f in file_names if f.endswith('_label.nii.gz')]
        label_file_names.sort()
        label_file_names = label_file_names
        data_file_names = [f.replace('_label', '') for f in label_file_names]
        file_num = len(data_file_names)
        dataset_coors = []  # shape = (dataset_size, VOI_nums, 3)
        images = []
        labels = []
        edges = []
        fissure_pros = []
        test_names = []
        for data_name, label_name in zip(data_file_names, label_file_names):
            print (data_name)
            # if data_name.endswith("7734.nii.gz"):
            #     print (data_name, "pass")
            #     continue
            test_names.append(data_name)
            itk_img = sitk.ReadImage(os.path.join(self.test_path, data_name))
            img = sitk.GetArrayFromImage(itk_img)
            itk_label = sitk.ReadImage(os.path.join(self.test_path, label_name))
            label = sitk.GetArrayFromImage(itk_label)
            edge_name = label_name.replace('_label', '_edge2')
            itk_edge = sitk.ReadImage(os.path.join(self.test_path, edge_name))
            edge = sitk.GetArrayFromImage(itk_edge)
            fissure_pro_name = label_name.replace('_label', '_fissure_prob')
            itk_fissure_pro = sitk.ReadImage(os.path.join(self.test_path, fissure_pro_name))
            fissure_pro = sitk.GetArrayFromImage(itk_fissure_pro)
            edge = self.one_hot(edge, 2)
            label = self.one_hot(label, self.num_classes)
            img = img.astype(np.float32)
            img = img.astype(np.float32) / 255.0
            label = label.astype(np.float32)
            edge = edge.astype(np.float32)
            fissure_pro = fissure_pro.astype(np.float32)
            fissure_pro = fissure_pro[..., np.newaxis]
            # label = np.pad(label, 1, 'constant', constant_values=0)

            img = img[..., np.newaxis]
            # label = label[..., np.newaxis]
            images.append(img)
            labels.append(label)
            edges.append(edge)
            fissure_pros.append(fissure_pro)
            coors = sliding_window(img.shape, self.window_size, self.test_moving_size)
            dataset_coors.append(coors)
        return (images,fissure_pros), labels, edges, dataset_coors, test_names
