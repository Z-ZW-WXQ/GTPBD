import os
import numpy as np
from PIL import Image

def compute_metrics(pred, label, th):
    """根据阈值计算评估指标"""
    pred_bin = (pred >= th).astype(int)
    tp = np.sum((pred_bin == 1) & (label == 1))
    fp = np.sum((pred_bin == 1) & (label == 0))
    fn = np.sum((pred_bin == 0) & (label == 1))
    return tp, tp + fn, tp + fp, tp + fp  # cntR, sumR, cntP, sumP

def generate_real_ev1_files(exp_dir, model_alphas, test_image_list, label_dir):
    """
    基于真实预测数据生成评估文件
    :param label_dir: 标签目录路径
    """
    for alpha in model_alphas:
        # 构建预测图目录路径
        pred_dir = os.path.join(exp_dir, str(alpha), "png")
        nms_eval_dir = os.path.join(exp_dir, str(alpha), "nms-eval")
        os.makedirs(nms_eval_dir, exist_ok=True)

        for img_name in test_image_list:
            # 加载预测图
            pred_path = os.path.join(pred_dir, img_name)
            pred = np.array(Image.open(pred_path).convert('L')) / 255.0
            
            # 加载标签
            label_path = os.path.join(label_dir, img_name)
            label = (np.array(Image.open(label_path).convert('L')) > 0).astype(int)
            
            # 验证尺寸一致性
            assert pred.shape == label.shape, f"尺寸不匹配: {img_name}"
            
            # 生成评估文件
            base_name = os.path.splitext(img_name)[0]
            ev1_file = os.path.join(nms_eval_dir, f"{base_name}_ev1.txt")
            
            with open(ev1_file, 'w') as f:
                for th in np.arange(0, 1.0, 0.01):
                    cntR, sumR, cntP, sumP = compute_metrics(pred, label, th)
                    f.write(f"{th:.2f}\t{cntR}\t{sumR}\t{cntP}\t{sumP}\n")
            print(f"Generated: {ev1_file}")

if __name__ == "__main__":
    # 配置参数
    exp_dir = "/data1/drive/data_nips/MUGE/tmp/trainvalsigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19"
    label_dir = "/data1/drive/data_nips/biaozhu_chongqing/chongqing_512_boundary/labels/test_png"
    model_alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    # 加载测试图片列表
    test_image_list = []
    with open('test_ois_list.txt', 'r') as f:
        test_image_list = [line.strip() for line in f if line.strip()]
    
    # 执行生成
    generate_real_ev1_files(exp_dir, model_alphas, test_image_list, label_dir)