import torch
import numpy as np

def fast_hist(a, b, n):
    # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)
    k = (a >= 0) & (a < n)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # 返回中，写对角线上的为分类正确的像素点
    return torch.bincount(n * a[k].to(torch.int) + b[k], minlength=n**2).reshape(n, n)  

def per_class_iou(hist):
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist)) 

def per_class_map(hist):
    return torch.diag(hist) / hist.sum(0) 

def per_class_recall(hist):
    return torch.diag(hist) / hist.sum(1) 

def mIoU_and_mAP(mask_pred, mask_true):
    """
    mask_true: one-hot型的mask
    """
    # 将模型预测概率转换为模型预测结果
    masks_result = torch.argmax(mask_pred, dim=1).float()
    mIoU = 0
    mAP = 0
    Recall = 0
    mIoU_list = torch.zeros([mask_pred.shape[1]], device=mask_pred.device)
    mAP_list = torch.zeros([mask_pred.shape[1]], device=mask_pred.device)
    Recall_list = torch.zeros([mask_pred.shape[1]], device=mask_pred.device)
    for i in range(masks_result.shape[0]):
        hist = fast_hist(masks_result[i], mask_true[i], mask_pred.shape[1])
        mious = per_class_iou(hist)
        maps = per_class_map(hist)
        recalls = per_class_recall(hist)
            
        mious_nan_pos = torch.isnan(mious)
        mious_n_pos = ~mious_nan_pos
        mious[mious_nan_pos] = 0

        maps_nan_pos = torch.isnan(maps)
        maps_n_pos = ~maps_nan_pos
        maps[maps_nan_pos] = 0

        recalls_nan_pos = torch.isnan(recalls)
        recalls_n_pos = ~recalls_nan_pos
        recalls[recalls_nan_pos] = 0

        mIoU_list += mious
        mIoU += torch.mean(mious[mious_n_pos])
        mAP_list += maps
        mAP += torch.mean(maps[maps_n_pos])
        Recall_list += recalls
        Recall += torch.mean(recalls[recalls_n_pos])
    return mIoU / masks_result.shape[0], \
           mIoU_list.cpu() / masks_result.shape[0], \
           mAP / masks_result.shape[0], \
           mAP_list.cpu() / masks_result.shape[0], \
           Recall / masks_result.shape[0], \
           Recall_list.cpu() / masks_result.shape[0], \

def IoU_and_AP(mask_pred, mask_true):
    """
    mask_true: one-hot型的mask
    """
    # 将模型预测概率转换为模型预测结果\
    mIoU = 0
    mAP = 0
    precision = torch.zeros(2, device=mask_pred.device)
    recall = torch.zeros(2, device=mask_pred.device)

    for i in range(mask_pred.shape[0]):
        hist = fast_hist(mask_pred[i], mask_true[i], 2)
        if hist.sum()-hist[0][0] == 0:
            mIoU += 1
        else:
            mIoU += torch.diag(hist)[1] / (hist.sum()-hist[0][0])
        mAP += torch.diag(hist).sum() / hist.sum()
        tmp_precision = per_class_map(hist)
        tmp_precision[torch.isnan(tmp_precision)] = 1
        precision += tmp_precision
        tmp_recall = torch.diag(hist) / hist.sum(1)
        tmp_recall[torch.isnan(tmp_recall)] = 1
        recall += tmp_recall


    return mIoU / mask_pred.shape[0], mAP / mask_pred.shape[0], precision / mask_pred.shape[0], recall / mask_pred.shape[0]




if __name__ == "__main__":
    pred = torch.tensor([[[[0.5, 0.8, 0.9],
                          [0.6, 0.1, 0.2],
                          [0.3, 0.2, 0.2]],

                          [[0.32, 0.234, 0.4],
                          [0.4, 0.67, 0.23],
                          [0.1, 0.23, 0.34]],

                          [[1.5, 2.8, 0.1],
                          [0.763534, 0.64, 0.7564],
                          [0.325, 0.3421, 0.1]]]])
    gt = torch.tensor([[[1, 1, 0],
                       [2, 1, 0],
                       [0, 0, 0]]])

    miou, miou_list = mIoU_and_mAP(pred, gt)





