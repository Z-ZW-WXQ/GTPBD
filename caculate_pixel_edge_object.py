import os
import numpy as np
from skimage import io, measure, segmentation
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import rasterio

# ================================= 路径配置 =================================
pred_path = "work_dirs/local-basic/260203_2009_gtaHR2csHR_mic_hrda_s2_ff667/SW_to_EC_maskradio_04_fuse"
gt_path = "BB_domain/EC/test/parcel_labels"

# ============================= 像素级指标计算 (多类别) ==============================
def compute_pixel_metrics_multiclass(pred, gt, num_classes=2):
    """
    计算多类别（包括0和1）的单图像混淆矩阵元素
    返回: 
        list_of_tp: 每个类别的真正例数列表
        list_of_fp: 每个类别的假正例数列表
        list_of_fn: 每个类别的假负例数列表
    """
    list_of_tp = []
    list_of_fp = []
    list_of_fn = []
    
    for cls in range(num_classes):
        # 对于当前类别cls，预测为cls的为正，其他为负
        tp = np.sum((pred == cls) & (gt == cls))
        fp = np.sum((pred == cls) & (gt != cls))
        fn = np.sum((pred != cls) & (gt == cls))
        list_of_tp.append(tp)
        list_of_fp.append(fp)
        list_of_fn.append(fn)
    
    return list_of_tp, list_of_fp, list_of_fn

# =========================== 对象级几何指标计算 =============================
def compute_object_metrics(pred_bin, gt_bin):
    """按面积加权计算过分割、欠分割误差（符合公式22-24）"""
    # 连通区域分析
    gt_label = measure.label(gt_bin, connectivity=2)
    pred_label = measure.label(pred_bin, connectivity=2)
    
    gt_regions = measure.regionprops(gt_label)
    pred_regions = measure.regionprops(pred_label)
    
    oc_list, uc_list, tc_list = [], [], []
    area_weights = []  # 用于存储每个预测区域的面积权重
    
    # 总预测区域面积（用于归一化）
    total_pred_area = sum(pred.area for pred in pred_regions) if pred_regions else 1e-10
    
    # 遍历每个预测对象
    for pred_region in pred_regions:
        pred_mask = pred_label == pred_region.label
        overlap_gt = gt_label[pred_mask]
        unique_gt = np.unique(overlap_gt[overlap_gt != 0])
        
        if len(unique_gt) == 0:
            oc = 1.0
            uc = 1.0
        else:
            # 找到重叠最大的真实对象
            max_gt_id = unique_gt[np.argmax([np.sum(overlap_gt == uid) for uid in unique_gt])]
            gt_region = gt_regions[max_gt_id - 1]  # label从1开始，索引需减1
            
            area_pred = pred_region.area
            area_gt = gt_region.area
            area_overlap = np.sum(pred_mask & (gt_label == max_gt_id))
            
            oc = 1 - area_overlap / area_gt
            uc = 1 - area_overlap / area_pred
        
        # tc = np.sqrt(oc * uc) #几何平均
        tc = np.sqrt((oc**2 + uc**2) / 2)  # 平方平均 (RMS)
        oc_list.append(oc)
        uc_list.append(uc)
        tc_list.append(tc)
        area_weights.append(pred_region.area / total_pred_area)
    
    # 按面积加权计算全局误差（公式22-24）
    GOC = np.sum(np.array(oc_list) * np.array(area_weights)) if pred_regions else 0
    GUC = np.sum(np.array(uc_list) * np.array(area_weights)) if pred_regions else 0
    GTC = np.sum(np.array(tc_list) * np.array(area_weights)) if pred_regions else 0
    
    return GOC, GUC, GTC

# =========================== 边界指标计算 =============================
def compute_boundary_metrics(gt_bin, pred_bin, thresholds=np.linspace(1, 5, 5)):
    """
    计算边界匹配指标（OIS和ODS）
    
    参数:
        gt_bin: 二值化的真实标签
        pred_bin: 二值化的预测结果
        thresholds: 距离阈值列表
        
    返回:
        ois: 最佳间隔分数 (Optimal Image Scale)
        ods: 最佳数据集分数 (Optimal Dataset Scale)
    """
    # 提取边界
    gt_boundary = segmentation.find_boundaries(gt_bin, mode='inner').astype(np.uint8)
    pred_boundary = segmentation.find_boundaries(pred_bin, mode='inner').astype(np.uint8)
    
    # 检查是否存在边界
    if np.sum(gt_boundary) == 0 and np.sum(pred_boundary) == 0:
        return 0.0, 0.0
    
    # 计算距离变换
    gt_dist = distance_transform_edt(1 - gt_boundary)
    pred_dist = distance_transform_edt(1 - pred_boundary)
    
    f1_scores = []
    for thresh in thresholds:
        # 匹配预测边界到真实边界
        match_pred = pred_boundary & (gt_dist <= thresh)
        match_gt = gt_boundary & (pred_dist <= thresh)
        
        # 计算TP, FP, FN
        tp = np.sum(match_pred)
        fp = np.sum(pred_boundary) - tp
        fn = np.sum(gt_boundary) - np.sum(match_gt)
        
        # 计算Precision, Recall, F1
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)
    
    # 计算OIS和ODS
    ois = np.max(f1_scores) if f1_scores else 0
    ods = np.mean(f1_scores) if f1_scores else 0
    
    return ois, ods

def read_tiff(fileName):
    with rasterio.open(fileName) as src:
        data = src.read()
    return data

# =============================== 主评估流程 ================================
if __name__ == "__main__":
    num_classes = 2 # 类别数，0和1
    
    # 初始化多类别像素级指标统计量
    total_tp_per_class = [0] * num_classes
    total_fp_per_class = [0] * num_classes
    total_fn_per_class = [0] * num_classes
    
    # 初始化二值像素级指标（为了计算Precision, Recall, F1, OA）
    total_tp_binary = 0   # 二值情况下的TP (通常指前景类别1)
    total_fp_binary = 0   # 二值情况下的FP
    total_fn_binary = 0   # 二值情况下的FN
    total_tn_binary = 0   # 二值情况下的TN

    # 初始化对象级指标统计量
    total_goc = total_guc = total_gtc = 0.0
    
    # 初始化边界指标统计量
    total_ois = total_ods = 0.0
    
    num_images = 0
    
    # 遍历所有预测文件，限制最大处理数量
    max_images = 1000
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.tif')]
    pred_files = pred_files[:max_images]

    for filename in tqdm(pred_files, desc="Processing Images"):
        gt_file_path = os.path.join(gt_path, filename)
        if not os.path.exists(gt_file_path):
            print(f"Skipping {filename}: Ground truth file not found.")
            continue
        
        pred = read_tiff(os.path.join(pred_path, filename))  
        gt = read_tiff(os.path.join(gt_path, filename)) 
        
        # 确保数据是整数类型且为单通道（H, W）
        if pred.ndim == 3:
            pred = pred.squeeze(0) # 去除可能的单通道维度
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)
        
        # 计算多类别像素指标 (用于mIoU和mAcc)
        tp_per_class, fp_per_class, fn_per_class = compute_pixel_metrics_multiclass(pred, gt, num_classes)
        
        # 累加每个类别的 TP, FP, FN (用于mIoU和mAcc)
        for cls in range(num_classes):
            total_tp_per_class[cls] += tp_per_class[cls]
            total_fp_per_class[cls] += fp_per_class[cls]
            total_fn_per_class[cls] += fn_per_class[cls]
        
        # 计算二值指标 (以类别1为前景，用于Precision, Recall, F1, OA)
        pred_binary = (pred == 1).astype(np.uint8)
        gt_binary = (gt == 1).astype(np.uint8)
        tp_b = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp_b = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn_b = np.sum((pred_binary == 0) & (gt_binary == 1))
        tn_b = np.sum((pred_binary == 0) & (gt_binary == 0))
        total_tp_binary += tp_b
        total_fp_binary += fp_b
        total_fn_binary += fn_b
        total_tn_binary += tn_b
        
        # 对象级指标（针对二值边缘/对象，通常仍以1为前景计算）
        GOC, GUC, GTC = compute_object_metrics(pred_binary, gt_binary) # 使用二值图
        total_goc += GOC
        total_guc += GUC
        total_gtc += GTC
        
        # 边界指标计算（新增）
        ois, ods = compute_boundary_metrics(gt_binary, pred_binary)
        total_ois += ois
        total_ods += ods
        
        num_images += 1

    # ============================= 指标计算 ================================
    # 1. 计算二值指标 (Precision, Recall, IoU, OA, F1) - 这些指标通常针对前景(1)计算
    P = total_tp_binary / (total_tp_binary + total_fp_binary + 1e-10)
    R = total_tp_binary / (total_tp_binary + total_fn_binary + 1e-10)
    IoU_binary = total_tp_binary / (total_tp_binary + total_fn_binary + total_fp_binary + 1e-10)
    OA_binary = (total_tp_binary + total_tn_binary) / (total_tp_binary + total_tn_binary + total_fp_binary + total_fn_binary + 1e-10)
    F1 = 2 * (P * R) / (P + R + 1e-10)

    # 2. 计算多类别指标 (mIoU 和 mAcc)
    ious_per_class = []
    accs_per_class = [] # 各类别的准确率 (Recall)
    
    for cls in range(num_classes):
        tp = total_tp_per_class[cls]
        fp = total_fp_per_class[cls]
        fn = total_fn_per_class[cls]
        # IoU for this class
        iou = tp / (tp + fp + fn + 1e-10)
        ious_per_class.append(iou)
        # Accuracy (Recall) for this class
        acc = tp / (tp + fn + 1e-10)
        accs_per_class.append(acc)
    
    mIoU = np.mean(ious_per_class) # 平均交并比
    mAcc = np.mean(accs_per_class) # 平均准确率 (宏平均)

    # 3. 对象级指标
    GOC = total_goc / num_images if num_images > 0 else 0
    GUC = total_guc / num_images if num_images > 0 else 0
    GTC = total_gtc / num_images if num_images > 0 else 0
    
    # 4. 边界指标（新增）
    OIS = total_ois / num_images if num_images > 0 else 0
    ODS = total_ods / num_images if num_images > 0 else 0

    # ============================== 结果输出 ================================
    print(f"\n{' Pixel-Level Metrics ':=^40}")
    print(f"Precision (P): {P:.4f}")       # 二值 Precision
    print(f"Recall (R): {R:.4f}")           # 二值 Recall
    print(f"F1-Score: {F1:.4f}")            # 二值 F1
    print(f"Overall Accuracy (OA): {OA_binary:.4f}") # 二值 OA
    print(f"mIoU: {mIoU:.4f}")              # 多类别 mIoU (替换原来的二值IoU)
    print(f"mAcc: {mAcc:.4f}")              # 多类别 mAcc (新增)

    print(f"\n{' Object Metrics ':=^40}")
    print(f"Global Over-segmentation Error (GOC): {GOC:.4f}")
    print(f"Global Under-segmentation Error (GUC): {GUC:.4f}")
    print(f"Global Total Classification Error (GTC): {GTC:.4f}")
    
    print(f"\n{' Boundary Metrics ':=^40}")
    print(f"Optimal Image Scale (OIS): {OIS:.4f}")
    print(f"Optimal Dataset Scale (ODS): {ODS:.4f}")
