import numpy as np
from scipy.ndimage import label, generate_binary_structure
import warnings

def extract_components(binary_mask, min_size=10):
    """提取连通组件，可过滤小组件"""
    structure = generate_binary_structure(2, 2)  # 8-邻域
    labeled_mask, num_components = label(binary_mask, structure=structure)
    
    components = []
    component_sizes = []
    
    for i in range(1, num_components + 1):
        component_mask = (labeled_mask == i).astype(np.uint8)
        component_size = np.sum(component_mask)
        
        # 过滤小组件
        if component_size >= min_size:
            components.append(component_mask)
            component_sizes.append(component_size)
    
    return components, labeled_mask, component_sizes

def compute_sIoU(gt_component, gt_labeled, pred_components, pred_labeled, debug=False):
    """
    计算sIoU(k) = |k ∩ ˆK(k)| / |k ∩ ˆK(k) \ 𝒜(k)|
    
    关键修改：当预测组件过大时，sIoU不应该为1
    """
    try:
        # 1. 找到所有与真实组件k相交的预测组件
        intersecting_preds = []
        intersecting_pred_indices = []
        
        # 获取与gt_component相交的预测组件
        intersection_mask = (gt_component > 0) & (pred_labeled > 0)
        
        if not np.any(intersection_mask):
            return 0.0
        
        # 获取相交的预测组件标签
        intersecting_labels = np.unique(pred_labeled[intersection_mask])
        intersecting_labels = intersecting_labels[intersecting_labels > 0]
        
        if len(intersecting_labels) == 0:
            return 0.0
        
        for pred_label in intersecting_labels:
            pred_component = (pred_labeled == pred_label).astype(np.uint8)
            intersecting_preds.append(pred_component)
            intersecting_pred_indices.append(pred_label)
        
        # 2. 计算ˆK(k)：所有相交预测组件的并集
        K_hat = np.zeros_like(gt_component, dtype=np.uint8)
        for pred in intersecting_preds:
            K_hat = np.logical_or(K_hat, pred)
        
        # 3. 计算交集：k ∩ ˆK(k)
        intersection = np.logical_and(gt_component, K_hat).astype(np.float32)
        intersection_area = np.sum(intersection)
        
        # 如果交集面积为0，返回0
        if intersection_area == 0:
            return 0.0
        
        # 4. 计算真实组件k的面积
        gt_area = np.sum(gt_component)
        
        # 5. 计算调整项𝒜(k)
        adjustment = np.zeros_like(gt_component, dtype=np.float32)
        
        # 获取当前组件的标签
        current_component_labels = np.unique(gt_labeled[gt_component > 0])
        if len(current_component_labels) == 0:
            return 0.0
        current_label = current_component_labels[0]
        
        # 获取所有其他真实组件标签
        all_gt_labels = np.unique(gt_labeled)
        all_gt_labels = all_gt_labels[all_gt_labels > 0]
        
        for other_gt_label in all_gt_labels:
            if other_gt_label == current_label:
                continue
                
            other_gt_component = (gt_labeled == other_gt_label).astype(np.uint8)
            other_intersection = (other_gt_component > 0) & (pred_labeled > 0)
            
            if not np.any(other_intersection):
                continue
            
            other_pred_labels = np.unique(pred_labeled[other_intersection])
            other_pred_labels = other_pred_labels[other_pred_labels > 0]
            
            for pred_label in other_pred_labels:
                if pred_label in intersecting_pred_indices:
                    pred_component = (pred_labeled == pred_label).astype(np.uint8)
                    
                    triple_intersection = np.logical_and(
                        np.logical_and(pred_component, gt_component),
                        other_gt_component
                    ).astype(np.float32)
                    
                    adjustment = np.logical_or(adjustment, triple_intersection)
        
        # 6. 从交集中减去调整项
        adjustment_area = np.sum(adjustment)
        denominator = intersection_area - adjustment_area
        
        # 关键修改：确保分母不为0且不超过交集
        if denominator <= 0:
            # 如果调整项过大，说明预测组件与多个真实组件重叠严重
            # 这种情况下，sIoU应该降低
            return max(0.0, intersection_area / gt_area * 0.5)
        
        # 7. 计算sIoU，但考虑预测组件过大的情况
        sIoU = intersection_area / denominator
        
        # 关键修改：如果预测组件过大，sIoU应该惩罚
        # 计算预测组件的总面积
        total_pred_area = 0
        for pred in intersecting_preds:
            total_pred_area += np.sum(pred)
        
        # 如果预测组件面积远大于真实组件，进行惩罚
        if total_pred_area > gt_area * 5:  # 预测组件大于真实组件5倍
            # 使用惩罚因子
            penalty_factor = min(1.0, gt_area * 5 / total_pred_area)
            sIoU = sIoU * penalty_factor
        
        # 确保sIoU在合理范围内
        sIoU = min(sIoU, 1.0)
        
        return float(sIoU)
    
    except Exception as e:
        if debug:
            print(f"    sIoU计算错误: {e}")
        return 0.0

def compute_component_metrics(predictions, ground_truths, threshold=0.5, iou_threshold=0.5, debug=False):
    """
    计算组件级指标：sIoU, PPV, F1*
    简化日志输出
    """
    all_TP = 0
    all_FN = 0
    all_FP = 0
    all_sIoU_values = []
    
    total_gt_pixels = 0
    total_pred_pixels = 0
    
    for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        try:
            # 确保数据是二维的
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            if gt.ndim == 3 and gt.shape[0] == 1:
                gt = gt.squeeze(0)
            
            # 二值化预测
            pred_binary = (pred > threshold).astype(np.uint8)
            gt_binary = gt.astype(np.uint8)
            
            # 统计像素数
            total_gt_pixels += np.sum(gt_binary)
            total_pred_pixels += np.sum(pred_binary)
            
            # 提取连通组件，过滤小组件
            gt_components, gt_labeled, gt_sizes = extract_components(gt_binary, min_size=10)
            pred_components, pred_labeled, pred_sizes = extract_components(pred_binary, min_size=10)
            
            # 1. 对于每个真实组件k，计算sIoU(k)
            gt_labels = np.unique(gt_labeled)
            gt_labels = gt_labels[gt_labels > 0]
            
            TP_per_sample = 0
            FN_per_sample = 0
            
            for gt_label in gt_labels:
                gt_component = (gt_labeled == gt_label).astype(np.uint8)
                sIoU = compute_sIoU(gt_component, gt_labeled, pred_components, pred_labeled)
                
                if sIoU > iou_threshold:
                    TP_per_sample += 1
                    all_sIoU_values.append(sIoU)
                else:
                    FN_per_sample += 1
            
            # 2. 对于每个预测组件ˆk，计算PPV(ˆk)
            pred_labels = np.unique(pred_labeled)
            pred_labels = pred_labels[pred_labels > 0]
            
            FP_per_sample = 0
            
            for pred_label in pred_labels:
                pred_component = (pred_labeled == pred_label).astype(np.uint8)
                
                # 找到与预测组件ˆk相交的真实组件
                intersection_mask = (pred_component > 0) & (gt_labeled > 0)
                if not np.any(intersection_mask):
                    FP_per_sample += 1
                    continue
                
                # 获取相交的真实组件标签
                intersecting_gt_labels = np.unique(gt_labeled[intersection_mask])
                intersecting_gt_labels = intersecting_gt_labels[intersecting_gt_labels > 0]
                
                if len(intersecting_gt_labels) == 0:
                    FP_per_sample += 1
                    continue
                
                # 计算ˆK(k)：所有相交真实组件的并集
                K_hat = np.zeros_like(pred_component, dtype=np.uint8)
                for gt_label in intersecting_gt_labels:
                    gt_component = (gt_labeled == gt_label).astype(np.uint8)
                    K_hat = np.logical_or(K_hat, gt_component)
                
                # 计算ˆk ∩ ˆK(k)
                intersection = np.logical_and(pred_component, K_hat).astype(np.float32)
                
                # 计算PPV
                pred_area = np.sum(pred_component)
                intersection_area = np.sum(intersection)
                
                if pred_area == 0:
                    PPV = 0
                else:
                    PPV = intersection_area / pred_area
                
                if PPV <= iou_threshold:
                    FP_per_sample += 1
            
            all_TP += TP_per_sample
            all_FN += FN_per_sample
            all_FP += FP_per_sample
            
        except Exception as e:
            if debug:
                warnings.warn(f"样本 {idx} 处理错误: {e}")
            continue
    
    # 3. 计算总体指标
    # sIoU: 所有TP组件的sIoU平均值
    sIoU = np.mean(all_sIoU_values) if len(all_sIoU_values) > 0 else 0.0
    
    # PPV: TP / (TP + FP)
    PPV = all_TP / (all_TP + all_FP) if (all_TP + all_FP) > 0 else 0.0
    
    # F1*: 2TP / (2TP + FN + FP)
    denominator = 2 * all_TP + all_FN + all_FP
    F1_star = 2 * all_TP / denominator if denominator > 0 else 0.0
    
    metrics_dict = {
        'threshold': threshold,
        'sIoU': float(sIoU),
        'PPV': float(PPV),
        'F1_star': float(F1_star),
        'TP': int(all_TP),
        'FN': int(all_FN),
        'FP': int(all_FP),
        'matched_components': len(all_sIoU_values),
        'total_gt_pixels': int(total_gt_pixels),
        'total_pred_pixels': int(total_pred_pixels)
    }
    
    return metrics_dict

def compute_all_metrics(predictions, ground_truths, thresholds=None, iou_threshold=0.5):
    """
    在多个阈值上计算组件级指标
    简化日志输出
    """
    # 如果未提供阈值，则根据预测分数的范围动态生成
    if thresholds is None:
        # 获取所有预测分数的最小值和最大值
        all_predictions = np.concatenate([p.flatten() for p in predictions])
        all_predictions = all_predictions[np.isfinite(all_predictions)]
        
        if len(all_predictions) == 0:
            print("警告: 所有预测分数都是无效值")
            return {
                'F1_star': 0, 'sIoU': 0, 'PPV': 0, 'threshold': 0,
                'TP': 0, 'FN': 0, 'FP': 0, 'matched_components': 0
            }
        
        min_val = np.min(all_predictions)
        max_val = np.max(all_predictions)
        
        # 使用更有意义的阈值：关注高异常分数区域
        # 由于异常分数可能是负值，我们需要找到合适的范围
        
        # 获取分数分布的百分位数
        p10 = np.percentile(all_predictions, 10)
        p50 = np.percentile(all_predictions, 50)
        p90 = np.percentile(all_predictions, 90)
        
        # 生成阈值：从p90到p10，共11个点
        # 这样我们关注的是异常分数较高的区域
        thresholds = np.linspace(p90, p10, 11)
        
        print(f"预测分数范围: [{min_val:.4f}, {max_val:.4f}]")
        print(f"阈值生成: 从{p90:.4f}(90百分位)到{p10:.4f}(10百分位)")
    
    print(f"使用阈值数量: {len(thresholds)}")
    print(f"阈值范围: [{thresholds[0]:.4f}, {thresholds[-1]:.4f}]")
    
    best_metrics = {
        'F1_star': 0, 
        'sIoU': 0, 
        'PPV': 0, 
        'threshold': 0, 
        'TP': 0,
        'FN': 0,
        'FP': 0,
        'matched_components': 0
    }
    
    # 尝试每个阈值
    for i, threshold in enumerate(thresholds):
        try:
            threshold_val = float(threshold)
            
            # 跳过可能不合理的阈值
            if not np.isfinite(threshold_val):
                continue
            
            # 只显示简化的进度信息
            if i == 0 or i == len(thresholds)-1 or (i+1) % 3 == 0:
                print(f"测试阈值 {i+1}/{len(thresholds)}: {threshold_val:.4f}")
            
            metrics_dict = compute_component_metrics(
                predictions, ground_truths, threshold_val, iou_threshold=iou_threshold
            )
            
            # 记录最佳F1*分数
            if metrics_dict['F1_star'] > best_metrics['F1_star']:
                best_metrics = metrics_dict
                
        except Exception as e:
            print(f"阈值 {threshold:.4f} 计算失败: {e}")
            continue
    
    # 输出最佳结果的详细信息
    print(f"\n最佳阈值: {best_metrics['threshold']:.4f}")
    print(f"匹配组件数: {best_metrics['matched_components']}")
    print(f"TP: {best_metrics['TP']}, FP: {best_metrics['FP']}, FN: {best_metrics['FN']}")
    print(f"sIoU: {best_metrics['sIoU']:.4f}, PPV: {best_metrics['PPV']:.4f}, F1*: {best_metrics['F1_star']:.4f}")
    print(f"总真实异常像素: {best_metrics.get('total_gt_pixels', 'N/A')}")
    print(f"总预测异常像素: {best_metrics.get('total_pred_pixels', 'N/A')}")
    
    return best_metrics
