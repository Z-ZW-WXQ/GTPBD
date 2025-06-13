import os
import numpy as np
from skimage import io, measure
from tqdm import tqdm
import rasterio
# ================================= 路径配置 =================================
pred_path = "/data1/drive/data_nips/FDA/source_only/global_to_north"
gt_path = "/data1/drive/GTD_UDA/north/mask_labels"

# ============================= 像素级指标计算 ==============================
def compute_pixel_metrics(pred, gt):
    """计算单个图像的TP, TN, FP, FN"""
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, tn, fp, fn

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
        
        tc = np.sqrt(oc * uc)
        oc_list.append(oc)
        uc_list.append(uc)
        tc_list.append(tc)
        area_weights.append(pred_region.area / total_pred_area)
    
    # 按面积加权计算全局误差（公式22-24）
    GOC = np.sum(np.array(oc_list) * np.array(area_weights)) if pred_regions else 0
    GUC = np.sum(np.array(uc_list) * np.array(area_weights)) if pred_regions else 0
    GTC = np.sum(np.array(tc_list) * np.array(area_weights)) if pred_regions else 0
    
    return GOC, GUC, GTC

def read_tiff(fileName):
    with rasterio.open(fileName) as src:
        data = src.read()
    return data

# =============================== 主评估流程 ================================
if __name__ == "__main__":
    # 初始化统计量
    total_tp = total_tn = total_fp = total_fn = 0
    # all_oc, all_uc, all_tc = [], [], []
    # 初始化全局误差累加器
    total_goc = total_guc = total_gtc = 0.0
    num_images = 0
    # 遍历所有预测文件，限制最大处理数量
    max_images = 1000  # 设置最多处理1000张
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.tif')]
    pred_files = pred_files[:max_images]  # 截取前1000张

    # 遍历所有预测文件
    # pred_files = [f for f in os.listdir(pred_path) if f.endswith('.tif')]
    for filename in tqdm(pred_files, desc="Processing Images"):
        # 检查目标文件是否存在
        gt_file_path = os.path.join(gt_path, filename)
        if not os.path.exists(gt_file_path):
            print(f"Skipping {filename}: Ground truth file not found.")
            continue
        # 加载数据
        
        pred = read_tiff(os.path.join(pred_path, filename))  
        gt = read_tiff(os.path.join(gt_path, filename)) 
        
        # 二值化处理
        pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin = (gt > 0.5).astype(np.uint8)
        
        # 像素级指标
        tp, tn, fp, fn = compute_pixel_metrics(pred_bin, gt_bin)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        
        # 对象级指标（直接返回全局误差）
        GOC, GUC, GTC = compute_object_metrics(pred_bin, gt_bin)
        total_goc += GOC
        total_guc += GUC
        total_gtc += GTC
        num_images += 1

    # ============================= 指标计算 ================================
    # 像素级指标
    P = total_tp / (total_tp + total_fp + 1e-10)
    R = total_tp / (total_tp + total_fn + 1e-10)
    IoU = total_tp / (total_tp + total_fn + total_fp + 1e-10)
    OA = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-10)
    F1 = 2 * (P * R) / (P + R + 1e-10)

    # 对象级指标
    # 计算平均值（假设所有图像权重相等）
    GOC = total_goc / num_images if num_images > 0 else 0
    GUC = total_guc / num_images if num_images > 0 else 0
    GTC = total_gtc / num_images if num_images > 0 else 0

    # ============================== 结果输出 ================================
    print(f"\n{' Metric ':=^40}")
    print(f"Precision (P): {P:.4f}")
    print(f"Recall (R): {R:.4f}")
    print(f"IoU: {IoU:.4f}")
    print(f"Overall Accuracy (OA): {OA:.4f}")
    print(f"F1-Score: {F1:.4f}")
    print(f"\n{' Object Metrics ':=^40}")
    print(f"Global Over-segmentation Error (GOC): {GOC:.4f}")
    print(f"Global Under-segmentation Error (GUC): {GUC:.4f}")
    print(f"Global Total Classification Error (GTC): {GTC:.4f}")