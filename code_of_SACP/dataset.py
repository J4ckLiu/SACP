import numpy as np
import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils import data
from libs.utils import *



class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels):
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = self.samples[ID]
        y = self.labels[ID]
        return X, y

def build_dataloader(model_name, data_name, data_dir, gamma=0.5):
    if data_name == 'ip':
        data_dict = sio.loadmat(data_dir + '/ip/Indian_pines_corrected.mat')
        hsi = data_dict['indian_pines_corrected']
        gt_dict = sio.loadmat(data_dir + '/ip/Indian_pines_gt.mat')
        gt = gt_dict['indian_pines_gt']
        h, w, d= hsi.shape
    elif data_name == 'pu':
        data_dict = sio.loadmat(data_dir + '/pu/PaviaU.mat')
        hsi = data_dict['paviaU']
        gt_dict = sio.loadmat(data_dir + '/pu/PaviaU_gt.mat')
        gt = gt_dict['paviaU_gt']
        h, w, d= hsi.shape
    elif data_name == 'sa':
        data_dict = sio.loadmat(data_dir + '/sa/Salinas_corrected.mat')
        hsi = data_dict['salinas_corrected']
        gt_dict = sio.loadmat(data_dir + '/sa/Salinas_gt.mat')
        gt = gt_dict['salinas_gt']
        h, w, d= hsi.shape
    else:
        raise ValueError(data_name + " is not supported.")

    max_value = hsi.max()
    hsi = np.transpose(hsi, (2,0,1))
    hsi = hsi - np.mean(hsi, axis=(1,2), keepdims=True)
    hsi = hsi / max_value
    if model_name == "hybrid":
        hsi = np.transpose(hsi, (1,2,0))
        hsi = applyPCA(hsi, 30)
        hsi = np.transpose(hsi, (2,0,1))
        d = 30
    hsi = hsi.reshape(np.prod(hsi.shape[:1]),np.prod(hsi.shape[1:]))
    gt = gt.reshape(np.prod(gt.shape[:2]),)


    total_size = len(gt)
    train_size = 250
    if model_name == 'cnn1d':
        train_size = 500
    cal_size = int((total_size - train_size) * 0.5)
    test_size = total_size - train_size - cal_size
    whole_data = hsi.reshape(d, h, w)
    if model_name == 'hybrid':
        patch_len = 12
    elif model_name == 'cnn3d':
        patch_len = 2
    elif model_name == 'sstn':
        patch_len = 4
    if model_name != 'cnn1d':
        padded_data = zeroPadding_3D(whole_data, patch_len)
    train_indices_zero, temp_indices, _, _ = train_test_split(
        range(len(gt)),
        gt,
        stratify=gt,
        test_size=total_size-train_size,
        random_state=2
    )
    train_indices = np.array([i for i in train_indices_zero if gt[i]!=0])

    sub_gt = gt[temp_indices]
    test_sub_indices, cal_sub_indices, _, _ = train_test_split(
        range(len(sub_gt)),
        sub_gt,
        stratify=sub_gt,
        test_size=cal_size,
        random_state=2
    )

    test_indices_zero = np.array([temp_indices[i] for i in test_sub_indices])
    cal_indices_zero = np.array([temp_indices[i] for i in cal_sub_indices])

    test_indices = np.array([i for i in test_indices_zero if gt[i]!=0])
    cal_indices = np.array([i for i in cal_indices_zero if gt[i]!=0])

    if data_name == 'ip':
        for _ in range(2):
            for e, i in enumerate(cal_indices):
                if gt[i] == 1:
                    train_indices = np.append(train_indices, i)
                    cal_indices = np.delete(cal_indices, e)
                    break
                if gt[i] == 7:
                    train_indices = np.append(train_indices, i)
                    cal_indices = np.delete(cal_indices, e)
                    break
                if gt[i] == 9:
                    train_indices = np.append(train_indices, i)
                    cal_indices = np.delete(cal_indices, e)
                    break
        train_size += 6
        cal_size -= 6
    if model_name == 'cnn1d':
        train_data = np.zeros((train_size, d))
        test_data = np.zeros((test_size, d))
        cal_data = np.zeros((cal_size, d))

        for e, i in enumerate(train_indices):
            row = i // w
            col = i % w
            train_data[e] = whole_data[:, row, col]

        for e, i in enumerate(test_indices):
            row = i // w
            col = i % w
            test_data[e] = whole_data[:, row, col]

        for e, i in enumerate(cal_indices):
            row = i // w
            col = i % w
            cal_data[e] = whole_data[:, row, col]
    else:
        train_data = np.zeros((train_size, d, 2*patch_len + 1, 2*patch_len + 1))
        test_data = np.zeros((test_size, d, 2*patch_len + 1, 2*patch_len + 1))
        cal_data = np.zeros((cal_size, d, 2*patch_len + 1, 2*patch_len + 1))

        y_train = gt[train_indices] - 1
        y_test = gt[test_indices] - 1
        y_cal = gt[cal_indices] - 1

        train_assign = indexToAssignment(train_indices, patch_len, whole_data.shape[1], whole_data.shape[2])
        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, patch_len, train_assign[i][0], train_assign[i][1])
            
        test_assign = indexToAssignment(test_indices, patch_len, whole_data.shape[1], whole_data.shape[2])
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, patch_len, test_assign[i][0], test_assign[i][1])

        cal_assign = indexToAssignment(cal_indices, patch_len, whole_data.shape[1], whole_data.shape[2])
        for i in range(len(cal_assign)):
            cal_data[i] = selectNeighboringPatch(padded_data, patch_len, cal_assign[i][0], cal_assign[i][1])
        
        if model_name == 'hybrid' or model_name == 'cnn3d':
            train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
            test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2], test_data.shape[3]))
            cal_data = np.reshape(cal_data, (cal_data.shape[0], 1, cal_data.shape[1], cal_data.shape[2], cal_data.shape[3]))

    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1
    y_cal = gt[cal_indices] - 1
        
    training_set = HSIDataset(range(len(train_indices)), train_data, y_train)
    test_set = HSIDataset(range(len(test_indices)), test_data, y_test)
    cal_set = HSIDataset(range(len(cal_indices)), cal_data, y_cal)

    _ = torch.utils.data.DataLoader(training_set, batch_size=256, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)
    calibloader = torch.utils.data.DataLoader(cal_set, batch_size=256, shuffle=False, num_workers=8)

    rt_dict = {
        'calibloader' : calibloader,
        'testloader' : testloader,
        'h' : h,
        'w' : w,
        'cal_indices' : cal_indices,
        'test_indices' : test_indices
    }

    return rt_dict


