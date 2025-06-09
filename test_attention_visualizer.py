#!/usr/bin/env python3
"""
测试注意力可视化器的简化脚本
如果没有预训练模型，将使用随机初始化的权重进行演示
"""

import torch
import os
from attention_visualizer import AttentionVisualizer

def test_attention_visualizer():
    """测试注意力可视化器"""
    print("开始测试注意力可视化器...")
    
    # 检查是否存在最优模型路径
    optimal_model_path = "exp/experiments/efficient_ablation_20250609_101146/embed_dim_ablation_embed_dim_128/best_model.pth"
    
    if not os.path.exists(optimal_model_path):
        print(f"最优模型文件不存在: {optimal_model_path}")
        print("将使用随机初始化权重进行演示...")
        model_path = "dummy_model.pth"  # 使用占位符路径
    else:
        model_path = optimal_model_path
        print(f"找到最优模型: {model_path}")
    
    try:
        # 初始化可视化器
        visualizer = AttentionVisualizer(model_path)
        
        # 测试模型加载
        print("\n测试模型加载...")
        visualizer.load_optimal_model()
        print("✓ 模型加载成功")
        
        # 测试数据获取
        print("\n测试数据获取...")
        samples = visualizer.get_test_samples(num_samples=2)  # 只测试2个样本
        print(f"✓ 成功获取 {len(samples)} 个测试样本")
        
        # 测试前向传播
        print("\n测试前向传播和注意力提取...")
        sample_image, sample_label = samples[0]
        image_batch = sample_image.unsqueeze(0).to(visualizer.device)
        output, attention_maps = visualizer.forward_with_attention(image_batch)
        
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 注意力层数: {len(visualizer.attention_maps)}")
        for layer_name, attention in visualizer.attention_maps.items():
            print(f"  {layer_name}: {attention.shape}")
        
        # 测试单个可视化功能
        print("\n测试注意力模式可视化...")
        visualizer.visualize_attention_patterns(sample_image, sample_label, 0)
        print("✓ 注意力模式可视化完成")
        
        print("\n测试注意力头分析...")
        visualizer.analyze_attention_heads(sample_image, sample_label, 0)
        print("✓ 注意力头分析完成")
        
        print("\n测试层级注意力对比...")
        visualizer.create_layer_comparison(sample_image, sample_label, 0)
        print("✓ 层级注意力对比完成")
        
        print("\n测试统计分析...")
        visualizer.create_attention_statistics(samples)
        print("✓ 统计分析完成")
        
        print("\n测试报告生成...")
        visualizer._generate_analysis_report(samples)
        print("✓ 分析报告生成完成")
        
        print(f"\n🎉 所有测试通过！结果保存在: {visualizer.output_dir}")
        print("\n生成的文件:")
        if os.path.exists(visualizer.output_dir):
            for file in os.listdir(visualizer.output_dir):
                file_path = os.path.join(visualizer.output_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def run_full_analysis():
    """运行完整的注意力分析"""
    print("\n" + "="*50)
    print("运行完整的注意力分析...")
    print("="*50)
    
    # 检查模型路径
    optimal_model_path = "exp/experiments/efficient_ablation_20250609_101146/embed_dim_ablation_embed_dim_128/best_model.pth"
    
    if not os.path.exists(optimal_model_path):
        print(f"最优模型文件不存在: {optimal_model_path}")
        print("将使用随机初始化权重进行演示...")
        model_path = "dummy_model.pth"
    else:
        model_path = optimal_model_path
    
    try:
        visualizer = AttentionVisualizer(model_path)
        visualizer.run_comprehensive_analysis(model_path)
        print("\n🎉 完整分析完成！")
        
    except Exception as e:
        print(f"❌ 完整分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    print("Vision Transformer 注意力可视化测试")
    print("="*50)
    
    # 首先运行单元测试
    if test_attention_visualizer():
        print("\n单元测试通过，继续运行完整分析...")
        
        # 询问用户是否继续运行完整分析
        response = input("\n是否运行完整的注意力分析？(y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            run_full_analysis()
        else:
            print("跳过完整分析。")
    else:
        print("\n单元测试失败，请检查代码。") 