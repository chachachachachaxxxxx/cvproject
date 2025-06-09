#!/usr/bin/env python3
"""
Vision Transformer 注意力机制可视化演示脚本

这个脚本演示了如何使用注意力可视化工具来分析
Vision Transformer在MNIST数据集上的注意力模式。

使用方法:
    python demo_attention_visualization.py
"""

import os
import sys
from attention_visualizer import AttentionVisualizer

def main():
    print("🔍 Vision Transformer 注意力机制可视化演示")
    print("=" * 60)
    
    # 模型路径配置
    # 如果您有训练好的模型，请修改这个路径
    optimal_model_path = "exp/experiments/efficient_ablation_20250609_101146/embed_dim_ablation_embed_dim_128/best_model.pth"
    
    if os.path.exists(optimal_model_path):
        print(f"✅ 找到训练好的最优模型: {optimal_model_path}")
        model_path = optimal_model_path
    else:
        print(f"⚠️  最优模型未找到: {optimal_model_path}")
        print("💡 将使用随机初始化权重进行演示")
        model_path = "demo_model.pth"  # 占位符路径
    
    try:
        # 1. 初始化注意力可视化器
        print("\n🚀 初始化注意力可视化器...")
        visualizer = AttentionVisualizer(model_path, device='cuda')
        
        # 2. 加载模型
        print("\n📥 加载最优ViT模型配置...")
        visualizer.load_optimal_model()
        print("   - 嵌入维度: 128")
        print("   - 注意力头数: 4") 
        print("   - Transformer层数: 6")
        print("   - Patch大小: 7×7")
        
        # 3. 获取测试样本
        print("\n📊 获取MNIST测试样本...")
        samples = visualizer.get_test_samples(num_samples=3)  # 演示用3个样本
        print(f"   - 成功获取 {len(samples)} 个不同类别的样本")
        
        # 4. 演示单个样本的详细分析
        print("\n🔬 演示单个样本的详细注意力分析...")
        sample_image, sample_label = samples[0]
        
        print(f"   分析样本: 数字 {sample_label}")
        
        # 基本注意力模式
        print("   - 生成层级注意力热力图...")
        visualizer.visualize_attention_patterns(sample_image, sample_label, 0)
        
        # 注意力头分析
        print("   - 分析多头注意力专业化...")
        visualizer.analyze_attention_heads(sample_image, sample_label, 0)
        
        # 层级演化对比
        print("   - 创建层级注意力演化对比...")
        visualizer.create_layer_comparison(sample_image, sample_label, 0)
        
        # 5. 批量处理所有样本
        print(f"\n⚡ 批量处理所有 {len(samples)} 个样本...")
        for i, (image, label) in enumerate(samples[1:], 1):  # 跳过已处理的第一个
            print(f"   处理样本 {i+1}: 数字 {label}")
            visualizer.visualize_attention_patterns(image, label, i)
            visualizer.analyze_attention_heads(image, label, i)
            visualizer.create_layer_comparison(image, label, i)
        
        # 6. 统计分析
        print("\n📈 生成统计分析...")
        visualizer.create_attention_statistics(samples)
        
        # 7. 生成报告
        print("\n📝 生成分析报告...")
        visualizer._generate_analysis_report(samples)
        
        # 8. 显示结果
        print(f"\n🎉 分析完成！结果保存在: {visualizer.output_dir}")
        print("\n📂 生成的文件:")
        
        if os.path.exists(visualizer.output_dir):
            files = sorted(os.listdir(visualizer.output_dir))
            for file in files:
                file_path = os.path.join(visualizer.output_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_type = get_file_description(file)
                    print(f"   📄 {file:<35} ({file_size:>8,} bytes) - {file_type}")
        
        print("\n🔍 主要可视化内容:")
        print("   📊 attention_pattern_sample_*.png  - 每层注意力热力图对比")
        print("   🎯 attention_heads_sample_*.png    - 多头注意力专业化分析") 
        print("   📈 layer_comparison_sample_*.png   - 层级注意力演化过程")
        print("   📉 attention_statistics.png       - 注意力统计特征分析")
        print("   📋 attention_analysis_report.md   - 详细分析报告")
        
        print("\n💡 使用建议:")
        print("   1. 查看 attention_pattern_sample_*.png 了解模型关注的图像区域")
        print("   2. 分析 attention_heads_sample_*.png 理解多头注意力的分工")
        print("   3. 观察 layer_comparison_sample_*.png 理解特征提取的层级性")
        print("   4. 阅读 attention_analysis_report.md 获取详细分析结论")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_file_description(filename):
    """获取文件类型描述"""
    if filename.startswith('attention_pattern_'):
        return "注意力模式热力图"
    elif filename.startswith('attention_heads_'):
        return "多头注意力分析"
    elif filename.startswith('layer_comparison_'):
        return "层级注意力对比"
    elif filename == 'attention_statistics.png':
        return "注意力统计分析"
    elif filename == 'attention_analysis_report.md':
        return "详细分析报告"
    else:
        return "其他文件"

if __name__ == "__main__":
    print("Vision Transformer 注意力机制可视化演示")
    print("基于MNIST手写数字识别任务")
    print("-" * 60)
    
    success = main()
    
    if success:
        print("\n✅ 演示成功完成！")
        print("\n🔗 相关资源:")
        print("   - 完整项目文档: docs/vision_transformer_mnist.md")
        print("   - 使用说明: README_attention_visualization.md")
        print("   - 测试脚本: test_attention_visualizer.py")
    else:
        print("\n❌ 演示未能完成，请检查错误信息")
        sys.exit(1) 