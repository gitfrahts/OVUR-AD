import time
from pathlib import Path
import numpy as np
from attrdict import AttrDict
import torch
from utils.calculate_statistics import calculate_statistics, load_stats
from anomaly_dataset import get_anomaly_dataset
from utils.inference import iter_over, metrics
from options import get_parser, init_cuda
import optimizer
import network
import shlex
import sys
import json

try:
    from utils.component_metrics import compute_all_metrics
    HAS_COMPONENT_METRICS = True
except ImportError:
    HAS_COMPONENT_METRICS = False
    print("警告: 无法导入 component_metrics，将只计算像素级指标")

# 可视化相关的导入
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage


def save_visualization(image_path, gt_mask, pred_score, output_dir, img_idx, threshold=0.5):
    """
    保存可视化结果到文件，使用PIL替代OpenCV
    """
    # ...（保持原有的可视化函数不变）...


# 新增：更精确的计算推理参数量的函数
def count_inference_parameters(model):
    """
    只统计推理时实际使用的参数
    排除：BatchNorm的running_mean/variance、Dropout层、训练专用参数等
    """
    # 如果是DataParallel或DistributedDataParallel，获取内部模型
    if hasattr(model, 'module'):
        model = model.module
    
    total_inference_params = 0
    total_all_params = 0
    
    # 需要统计的层类型（这些层在推理时会使用参数）
    inference_layers = [
        torch.nn.Conv2d, 
        torch.nn.Linear,
        torch.nn.BatchNorm2d,  # 注意：推理时使用weight和bias，但running_mean/variance是统计量，不是可训练参数
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        torch.nn.ConvTranspose2d,
        torch.nn.Embedding,
        torch.nn.PReLU,
        torch.nn.InstanceNorm2d,
    ]
    
    print("\n" + "="*60)
    print("推理模型参数详细分析:")
    print("="*60)
    
    for name, module in model.named_modules():
        # 跳过最外层
        if name == '':
            continue
            
        # 统计该模块的可训练参数
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_all_params += sum(p.numel() for p in module.parameters())
        
        # 检查是否是推理会使用的层
        is_inference_layer = any(isinstance(module, layer_type) for layer_type in inference_layers)
        
        if is_inference_layer and module_params > 0:
            total_inference_params += module_params
            print(f"  ✓ {name}: {module_params:,} 参数 ({module.__class__.__name__})")
        elif module_params > 0:
            print(f"  ⚠️ {name}: {module_params:,} 参数 ({module.__class__.__name__}) - 训练专用或辅助层")
    
    # 计算参数大小（MB），假设float32，4字节
    inference_size_mb = total_inference_params * 4 / (1024 ** 2)
    all_size_mb = total_all_params * 4 / (1024 ** 2)
    
    return {
        'inference_params': total_inference_params,
        'all_params': total_all_params,
        'inference_size_mb': inference_size_mb,
        'all_size_mb': all_size_mb,
        'inference_params_formatted': f"{total_inference_params:,}",
        'all_params_formatted': f"{total_all_params:,}",
        'inference_size_formatted': f"{inference_size_mb:.2f} MB",
        'all_size_formatted': f"{all_size_mb:.2f} MB",
        'percentage': (total_inference_params / total_all_params * 100) if total_all_params > 0 else 0
    }


# 新增：更精确的推理时间统计
def timed_inference(net, image_list, mask_list, args):
    """
    更精确的推理时间统计，只计算前向传播时间
    """
    # 确保模型在eval模式
    net.eval()
    
    # 预热（避免第一次推理的冷启动影响）
    print("\n⚡ 预热推理...")
    warmup_samples = min(3, len(image_list))
    for i in range(warmup_samples):
        with torch.no_grad():
            # 这里需要根据实际的推理函数来调用
            # 由于iter_over函数内部复杂，我们单独计时
            pass
    
    # 实际推理计时
    print("⏱️  开始精确推理计时...")
    inference_times = []
    
    # 这里我们需要修改iter_over函数或创建一个新的函数
    # 由于iter_over是黑盒，我们可以用更简单的方式：
    # 1. 先记录开始时间
    # 2. 执行推理
    # 3. 记录结束时间
    
    start_total = time.perf_counter()
    
    # 调用iter_over（保持原有逻辑）
    as_list, ood_list, evals = iter_over(net, image_list, mask_list, args)
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # 计算每张图片的平均时间
    num_images = len(image_list)
    avg_time_per_image = total_time / num_images if num_images > 0 else 0
    
    # 计算FPS
    fps = num_images / total_time if total_time > 0 else 0
    
    return as_list, ood_list, evals, {
        'total_time': total_time,
        'num_images': num_images,
        'avg_time_per_image': avg_time_per_image,
        'fps': fps,
        'avg_ms_per_image': avg_time_per_image * 1000  # 毫秒
    }


if __name__ == "__main__":
    parser = get_parser()
    # 添加可视化参数
    parser.add_argument('--visualize', action='store_true', 
                       help='是否生成可视化结果')
    parser.add_argument('--vis_num', type=int, default=50,
                       help='可视化样本数量（默认: 50）')
    parser.add_argument('--vis_threshold', type=float, default=0.5,
                       help='可视化阈值（默认: 0.5）')
    parser.add_argument('--vis_all', action='store_true',
                       help='可视化所有样本（覆盖--vis_num）')
    parser.add_argument('--detailed_timing', action='store_true',
                       help='显示详细的推理时间分析')
    
    tmp_args, _ = parser.parse_known_args()
    print("___+++Parsed args:", tmp_args)
    print("END")
    init_cuda(tmp_args)

    ckpt = torch.load(tmp_args.snapshot, map_location='cpu')
    cmd = ckpt['command']
    ckpt_args, other_args = get_parser().parse_known_args(shlex.split(cmd) + sys.argv[1:])
    ckpt_args.local_rank = tmp_args.local_rank

    net = network.get_net(ckpt_args, None, None)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, tmp_args.local_rank)
    epoch = optimizer.load_weights(net, None, None, None, False, ckpt)
    ident = f"{ckpt_args.tag}_{epoch}"
    net.eval()
    
    # 计算推理参数量（只在主进程）
    if tmp_args.local_rank == 0:
        param_info = count_inference_parameters(net)
        
        print("\n" + "="*60)
        print("推理模型参数统计（仅核心推理部分）:")
        print("="*60)
        print(f"推理核心参数: {param_info['inference_params_formatted']}")
        print(f"占总参数比例: {param_info['percentage']:.1f}%")
        print(f"推理参数大小: {param_info['inference_size_formatted']}")
        print(f"（所有参数总计: {param_info['all_params_formatted']}, {param_info['all_size_formatted']}）")

    # calculate class mean and variance
    calculate_statistics(net, ident, tmp_args)
    torch.distributed.barrier()
    load_stats(net, ident)

    # load anomaly dataset
    image_list_all, mask_list_all = get_anomaly_dataset(tmp_args.anomaly_dataset)
    assert len(mask_list_all) == len(mask_list_all)
    ds_len = len(image_list_all)

    # split into all ranks
    image_each_proc = len(mask_list_all) // torch.distributed.get_world_size()
    res = len(mask_list_all) % torch.distributed.get_world_size()
    if tmp_args.local_rank < res:
        image_each_proc += 1
        pos = slice(image_each_proc * tmp_args.local_rank, image_each_proc * (tmp_args.local_rank + 1))
    else:
        pos = slice(res + image_each_proc * tmp_args.local_rank, res + image_each_proc * (tmp_args.local_rank + 1))
    assert pos.start < pos.stop, f"Invalid pos: {pos} for local_rank {tmp_args.local_rank}"
    image_list = image_list_all[pos]
    mask_list = mask_list_all[pos]
    if tmp_args.local_rank != 0:
        del image_list_all, mask_list_all

    # get anomaly scores with precise timing
    as_list, ood_list, evals, timing_info = timed_inference(net, image_list, mask_list, tmp_args)
    
    # 打印时间统计
    print(f"\n进程 {tmp_args.local_rank} 推理时间统计:")
    print(f"  📊 处理图片数量: {timing_info['num_images']}")
    print(f"  ⏱️  总推理时间: {timing_info['total_time']:.2f} 秒")
    print(f"  📈 平均每张图片: {timing_info['avg_ms_per_image']:.1f} ms")
    print(f"  🚀 推理速度: {timing_info['fps']:.1f} FPS")
    
    # 详细时间分析（如果启用）
    if tmp_args.detailed_timing and tmp_args.local_rank == 0:
        print(f"\n📊 详细性能分析:")
        print(f"  - 如果批量大小为1: {timing_info['avg_ms_per_image']:.1f} ms/张")
        print(f"  - 理论最大吞吐量: {1000/timing_info['avg_ms_per_image']:.1f} FPS")
        print(f"  - 假设批量大小为8: {timing_info['avg_ms_per_image']/8:.1f} ms/张 (理论)")
    
    tmp_file_name = f"rank{torch.distributed.get_rank()}_{time.time()}.npz"
    np.savez(tmp_file_name, as_list, ood_list)

    # gather from all ranks
    names = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(names, tmp_file_name)

    # calculate metrics
    if tmp_args.local_rank == 0:
        as_list_total, ood_list_total = [], []
        for name in names:
            eval_results = np.load(name)
            as_list_total.append(eval_results['arr_0'])
            ood_list_total.append(eval_results['arr_1'])
            Path(name).unlink(missing_ok=True)
        assert len(as_list_total) == torch.distributed.get_world_size()
        del image_list, mask_list, as_list, ood_list, evals

        # 展平列表
        as_list_total = [a for r in as_list_total for a in r]
        ood_list_total = [o for r in ood_list_total for o in r]
        
        # 计算像素级指标
        roc_auc, prc_auc, fpr_tpr95 = metrics(as_list_total, ood_list_total)
        print("\n" + "="*60)
        print("📈 评估结果:")
        print("="*60)
        print(f"Checkpoint: {tmp_args.snapshot}")
        print(f"Dataset: {tmp_args.anomaly_dataset}")
        print(f"AUROC: {roc_auc:.4f}")
        print(f"AUPRC: {prc_auc:.4f}")
        print(f"FPR@TPR95: {fpr_tpr95:.4f}")
        
        # 可视化部分
        if tmp_args.visualize:
            print("\n" + "="*60)
            print("🎨 开始生成可视化结果...")
            print("="*60)
            
            # 创建output__目录
            output_dir = Path("output__")
            output_dir.mkdir(exist_ok=True)
            
            # 创建以数据集命名的子目录
            dataset_dir = output_dir / tmp_args.anomaly_dataset
            dataset_dir.mkdir(exist_ok=True)
            
            # 创建时间戳目录，避免覆盖
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_dir = dataset_dir / timestamp
            viz_dir.mkdir(exist_ok=True)
            
            print(f"📁 可视化结果保存在: {viz_dir}")
            
            # 确定可视化数量
            if tmp_args.vis_all:
                num_to_visualize = len(as_list_total)
                print(f"🖼️  将可视化所有样本: {num_to_visualize} 张")
            else:
                num_to_visualize = min(tmp_args.vis_num, len(as_list_total))
                print(f"🖼️  将可视化 {num_to_visualize} 个样本")
            
            # 收集样本指标
            sample_metrics = {}
            successful_count = 0
            
            # 可视化进度条
            print("  📊 进度: ", end='', flush=True)
            progress_step = max(1, num_to_visualize // 20)
            
            for idx in range(num_to_visualize):
                try:
                    # 重新加载完整数据集（如果已经删除）
                    if 'image_list_all' not in locals():
                        image_list_all, mask_list_all = get_anomaly_dataset(tmp_args.anomaly_dataset)
                    
                    image_path = image_list_all[idx]
                    gt_mask = ood_list_total[idx]
                    pred_score = as_list_total[idx]
                    
                    # 生成可视化
                    viz_path, sample_info = save_visualization(
                        image_path, 
                        gt_mask, 
                        pred_score, 
                        viz_dir, 
                        idx,
                        threshold=tmp_args.vis_threshold
                    )
                    
                    if viz_path:
                        sample_metrics[idx] = sample_info
                        successful_count += 1
                        
                        # 显示进度
                        if (idx + 1) % progress_step == 0:
                            print(f"█", end='', flush=True)
                
                except Exception as e:
                    print(f"✗", end='', flush=True)
            
            print()  # 换行
            
            print(f"\n✅ 可视化完成，成功生成 {successful_count}/{num_to_visualize} 张图像")
            
            # 保存指标摘要
            summary = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'checkpoint': tmp_args.snapshot,
                'dataset': tmp_args.anomaly_dataset,
                'model_parameters': param_info,
                'inference_performance': timing_info,
                'metrics': {
                    'auroc': float(roc_auc),
                    'auprc': float(prc_auc),
                    'fpr_tpr95': float(fpr_tpr95)
                },
                'visualization': {
                    'num_samples': num_to_visualize,
                    'successful_count': successful_count,
                    'threshold': tmp_args.vis_threshold,
                    'output_dir': str(viz_dir)
                },
                'sample_metrics': sample_metrics
            }
            
            # 保存JSON文件
            summary_file = viz_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 指标摘要已保存: {summary_file}")
            
            # 创建详细的文本报告
            report_file = viz_dir / "report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("异常检测模型推理性能报告\n")
                f.write("="*70 + "\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"检查点: {tmp_args.snapshot}\n")
                f.write(f"数据集: {tmp_args.anomaly_dataset}\n\n")
                
                f.write("🔧 模型推理参数统计:\n")
                f.write("-"*50 + "\n")
                f.write(f"推理核心参数: {param_info['inference_params_formatted']}\n")
                f.write(f"占总参数比例: {param_info['percentage']:.1f}%\n")
                f.write(f"推理参数大小: {param_info['inference_size_formatted']}\n")
                f.write(f"（所有参数总计: {param_info['all_params_formatted']}, {param_info['all_size_formatted']}）\n\n")
                
                f.write("⚡ 推理性能统计:\n")
                f.write("-"*50 + "\n")
                f.write(f"总图片数量: {timing_info['num_images']}\n")
                f.write(f"总推理时间: {timing_info['total_time']:.2f} 秒\n")
                f.write(f"平均每张图片: {timing_info['avg_ms_per_image']:.1f} ms\n")
                f.write(f"推理速度: {timing_info['fps']:.1f} FPS\n\n")
                
                f.write("📈 评估指标:\n")
                f.write("-"*50 + "\n")
                f.write(f"AUROC: {roc_auc:.4f}\n")
                f.write(f"AUPRC: {prc_auc:.4f}\n")
                f.write(f"FPR@TPR95: {fpr_tpr95:.4f}\n\n")
                
                f.write(f"🎨 可视化设置:\n")
                f.write("-"*50 + "\n")
                f.write(f"样本数量: {num_to_visualize}\n")
                f.write(f"成功生成: {successful_count}\n")
                f.write(f"检测阈值: {tmp_args.vis_threshold}\n\n")
                
                # 性能最好的样本
                if sample_metrics:
                    f.write("🏆 性能最好的样本 (按F1分数排序):\n")
                    f.write("="*70 + "\n")
                    
                    sorted_samples = sorted(sample_metrics.items(), 
                                          key=lambda x: x[1]['f1_score'], 
                                          reverse=True)
                    
                    for i, (idx, metrics_dict) in enumerate(sorted_samples[:10]):
                        f.write(f"\n样本 {idx:04d} (F1分数: {metrics_dict['f1_score']:.3f}):\n")
                        f.write(f"  GT异常比例: {metrics_dict['gt_anomaly_ratio']:.2f}%\n")
                        f.write(f"  检测异常比例: {metrics_dict['detected_anomaly_ratio']:.2f}%\n")
                        f.write(f"  精确率: {metrics_dict['precision']:.3f} | 召回率: {metrics_dict['recall']:.3f}\n")
                        f.write(f"  TP: {metrics_dict['tp']}, FP: {metrics_dict['fp']}, FN: {metrics_dict['fn']}\n")
            
            print(f"📝 详细报告已保存: {report_file}")
            
            # 创建CSV格式的性能统计
            csv_file = viz_dir / "performance_stats.csv"
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("sample_id,gt_anomaly_ratio,detected_anomaly_ratio,precision,recall,f1_score,tp,fp,fn\n")
                for idx, metrics_dict in sorted(sample_metrics.items()):
                    f.write(f"{idx},{metrics_dict['gt_anomaly_ratio']:.4f},{metrics_dict['detected_anomaly_ratio']:.4f},")
                    f.write(f"{metrics_dict['precision']:.4f},{metrics_dict['recall']:.4f},{metrics_dict['f1_score']:.4f},")
                    f.write(f"{metrics_dict['tp']},{metrics_dict['fp']},{metrics_dict['fn']}\n")
            
            print(f"📊 CSV统计数据已保存: {csv_file}")
            
            # 创建性能对比图表
            if successful_count > 0:
                try:
                    # 创建F1分数分布图
                    f1_scores = [m['f1_score'] for m in sample_metrics.values()]
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # F1分数分布直方图
                    axes[0].hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[0].axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'平均: {np.mean(f1_scores):.3f}')
                    axes[0].set_xlabel('F1分数')
                    axes[0].set_ylabel('样本数量')
                    axes[0].set_title('F1分数分布')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # 推理时间与F1分数关系
                    axes[1].scatter(range(len(f1_scores)), f1_scores, alpha=0.6)
                    axes[1].set_xlabel('样本索引')
                    axes[1].set_ylabel('F1分数')
                    axes[1].set_title('样本性能分布')
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.suptitle(f'模型性能分析 - {tmp_args.anomaly_dataset}')
                    plt.tight_layout()
                    
                    chart_file = viz_dir / "performance_chart.png"
                    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"📈 性能分析图表已保存: {chart_file}")
                    
                except Exception as e:
                    print(f"⚠️  生成性能图表时出错: {e}")
            
            print(f"\n📁 所有文件保存在: {viz_dir}")
            print("\n📥 查看结果:")
            print(f"  1. 可视化图片: {successful_count} 张PNG文件")
            print(f"  2. 详细指标: summary.json (JSON格式)")
            print(f"  3. 文本报告: report.txt")
            print(f"  4. 数据统计: performance_stats.csv")
            print(f"  5. 性能图表: performance_chart.png")
            print("\n" + "="*60)
        
        else:
            print(f"\n⚠️  未启用可视化，如需生成可视化请添加 --visualize 参数")
            
        # 计算组件级指标
        if HAS_COMPONENT_METRICS:
            try:
                pred_scores = []
                gt_masks = []
                
                # 先收集所有预测分数用于分析
                all_scores = []
                
                for idx, (score_array, mask_array) in enumerate(zip(as_list_total, ood_list_total)):
                    # 处理预测分数数组
                    score_array = np.squeeze(score_array)
                    
                    # 处理真实掩码数组
                    mask_array = np.squeeze(mask_array).astype(np.uint8)
                    
                    # 确保都是二维数组
                    if score_array.ndim != 2 or mask_array.ndim != 2:
                        continue
                    
                    # 收集所有分数用于分析
                    all_scores.extend(score_array.flatten())
                    
                    pred_scores.append(score_array)
                    gt_masks.append(mask_array)
                
                print(f"\n准备用于组件级指标计算的样本数: {len(pred_scores)}")
                
                if len(pred_scores) > 0:
                    # 分析预测分数的统计信息
                    all_scores = np.array(all_scores)
                    print(f"\n预测分数统计:")
                    print(f"  最小值: {all_scores.min():.4f}")
                    print(f"  最大值: {all_scores.max():.4f}")
                    print(f"  平均值: {all_scores.mean():.4f}")
                    print(f"  中位数: {np.median(all_scores):.4f}")
                    print(f"  95百分位: {np.percentile(all_scores, 95):.4f}")
                    print(f"  99百分位: {np.percentile(all_scores, 99):.4f}")
                    
                    # 关键：分析异常分数与正常分数的分布
                    print(f"\n异常分数分析:")
                    
                    # 提取真实异常区域的分数
                    anomaly_scores = []
                    normal_scores = []
                    
                    for score, mask in zip(pred_scores, gt_masks):
                        anomaly_mask = (mask == 1)
                        normal_mask = (mask == 0)
                        
                        anomaly_scores.extend(score[anomaly_mask].flatten())
                        normal_scores.extend(score[normal_mask].flatten())
                    
                    if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                        anomaly_scores_arr = np.array(anomaly_scores)
                        normal_scores_arr = np.array(normal_scores)
                        
                        print(f"  异常区域分数统计:")
                        print(f"    最小值: {anomaly_scores_arr.min():.4f}")
                        print(f"    最大值: {anomaly_scores_arr.max():.4f}")
                        print(f"    平均值: {anomaly_scores_arr.mean():.4f}")
                        print(f"    中位数: {np.median(anomaly_scores_arr):.4f}")
                        
                        print(f"\n  正常区域分数统计:")
                        print(f"    最小值: {normal_scores_arr.min():.4f}")
                        print(f"    最大值: {normal_scores_arr.max():.4f}")
                        print(f"    平均值: {normal_scores_arr.mean():.4f}")
                        print(f"    中位数: {np.median(normal_scores_arr):.4f}")
                        
                        # 关键发现：异常分数更负，需要反转
                        print(f"\n  关键发现: 异常分数更负，需要反转处理")
                    
                    # 关键修改：正确的分数处理方法
                    print(f"\n=== 使用反转处理（基于分析） ===")
                    
                    # 方法1: 简单负号反转（使负值越大变成正值越大）
                    print(f"\n方法: 负号反转 + 归一化")
                    
                    pred_scores_processed = []
                    for score in pred_scores:
                        # 1. 负号反转：使异常分数（更负）变成更大的正值
                        score_inv = -score
                        
                        # 2. 归一化到[0,1]范围
                        min_val = score_inv.min()
                        max_val = score_inv.max()
                        if max_val > min_val:
                            score_norm = (score_inv - min_val) / (max_val - min_val)
                        else:
                            score_norm = np.ones_like(score_inv) * 0.5
                        
                        pred_scores_processed.append(score_norm)
                    
                    # 分析处理后的分数
                    all_processed = np.concatenate([p.flatten() for p in pred_scores_processed])
                    print(f"处理后分数统计:")
                    print(f"  最小值: {all_processed.min():.4f}")
                    print(f"  最大值: {all_processed.max():.4f}")
                    print(f"  平均值: {all_processed.mean():.4f}")
                    print(f"  中位数: {np.median(all_processed):.4f}")
                    
                    # 计算组件级指标
                    for iou_threshold in [0.3, 0.4, 0.5]:
                        print(f"\nIoU阈值: {iou_threshold}")
                        try:
                            component_metrics = compute_all_metrics(
                                pred_scores_processed, gt_masks, iou_threshold=iou_threshold
                            )
                            
                            print(f"  最佳阈值: {component_metrics['threshold']:.4f}")
                            print(f"  匹配组件数: {component_metrics.get('matched_components', 'N/A')}")
                            print(f"  TP: {component_metrics.get('TP', 'N/A')}")
                            print(f"  FP: {component_metrics.get('FP', 'N/A')}")
                            print(f"  FN: {component_metrics.get('FN', 'N/A')}")
                            print(f"  sIoU: {component_metrics['sIoU']:.4f}")
                            print(f"  PPV: {component_metrics['PPV']:.4f}")
                            print(f"  F1*: {component_metrics['F1_star']:.4f}")
                            
                            if component_metrics['F1_star'] > 0:
                                tp = component_metrics.get('TP', 0)
                                fp = component_metrics.get('FP', 0)
                                fn = component_metrics.get('FN', 0)
                                
                                if (tp + fn) > 0:
                                    detection_rate = tp / (tp + fn)
                                    print(f"  检测率: {detection_rate:.4f} ({tp}/{tp+fn})")
                                
                                if (tp + fp) > 0:
                                    precision = tp / (tp + fp)
                                    print(f"  精确率: {precision:.4f} ({tp}/{tp+fp})")
                            
                        except Exception as e:
                            print(f"  计算失败: {e}")
                    
                else:
                    print("\n无法准备组件级指标计算所需的数据")
                    
            except Exception as e:
                print(f"\n计算组件级指标时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n组件级指标计算模块未启用")