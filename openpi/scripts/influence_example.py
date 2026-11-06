#!/usr/bin/env python3
"""OpenPI LoRA Influence 计算的完整示例.

这个示例展示了如何:
1. 收集 eval rollouts
2. 运行 influence 计算
3. 分析结果并选择高质量数据

假设你已经有:
- 训练好的 OpenPI LoRA checkpoint
- 训练数据集
- 一些 eval rollouts
"""

import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def example_collect_rollouts():
    """示例: 如何在 rollout 过程中保存数据."""
    from save_eval_rollouts import EpisodeSaver
    
    # 创建 saver
    saver = EpisodeSaver("./eval_rollouts", format="dict")
    
    # 模拟 rollout (实际应该调用你的 policy server)
    for episode_id in range(10):
        # 模拟一个 episode
        episode_length = np.random.randint(20, 50)
        
        for t in range(episode_length):
            # 模拟观测
            observation = {
                "images": {
                    "base_0_rgb": np.random.rand(224, 224, 3),
                    "left_wrist_0_rgb": np.random.rand(224, 224, 3),
                    "right_wrist_0_rgb": np.random.rand(224, 224, 3),
                },
                "state": np.random.rand(14),  # 假设 state_dim=14
            }
            
            # 模拟动作
            action = np.random.rand(7)  # 假设 action_dim=7
            
            # 模拟 reward
            reward = 0.0 if t < episode_length - 1 else np.random.choice([0.0, 1.0])
            
            saver.add_step(observation, action, reward)
        
        # 保存 episode
        total_reward = 1.0 if np.random.rand() > 0.5 else 0.0
        saver.save_episode(reward=total_reward, episode_id=episode_id)
    
    print(f"已保存 {saver.episode_count} 个 eval episodes")


def example_run_influence():
    """示例: 运行 influence 计算."""
    import subprocess
    
    cmd = [
        "python", "openpi/scripts/influence_pi05_lora.py",
        "--config-name", "pi05_square_d1_lora",
        "--checkpoint-dir", "./checkpoints/pi05_square_d1_lora/exp1/20000",
        "--eval-rollouts-dir", "./eval_rollouts",
        "--output-path", "./results/influence.pkl",
        "--proj-dim", "4096",
        "--lissa-depth", "8",
        "--lissa-samples", "4",
        "--stage-cache-prefix", "./cache",
    ]
    
    print("运行 influence 计算...")
    print(" ".join(cmd))
    
    # 注释掉实际运行,避免在示例中执行
    # subprocess.run(cmd, check=True)
    
    print("计算完成!")


def example_analyze_results(result_path: str = "./results/influence.pkl"):
    """示例: 分析 influence 结果."""
    
    # 加载结果
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    
    influence_scores = result["influence_scores"]
    episode_rewards = result["metadata"]["episode_rewards"]
    
    print("=" * 60)
    print("Influence 分析结果")
    print("=" * 60)
    
    # 基本统计
    print(f"\n训练样本数: {len(influence_scores)}")
    print(f"Eval episodes: {len(episode_rewards)}")
    print(f"Eval 平均 reward: {episode_rewards.mean():.4f}")
    print(f"\nInfluence scores 统计:")
    print(f"  Mean: {influence_scores.mean():.6f}")
    print(f"  Std:  {influence_scores.std():.6f}")
    print(f"  Min:  {influence_scores.min():.6f}")
    print(f"  Max:  {influence_scores.max():.6f}")
    
    # Top-K 最有影响力的样本
    print("\n" + "=" * 60)
    print("Top 10 最有影响力的训练样本 (正向)")
    print("=" * 60)
    
    top_k = 10
    top_indices = np.argsort(influence_scores)[-top_k:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:2d}. Sample {idx:6d}: score = {influence_scores[idx]:10.6f}")
    
    # Top-K 负面影响最大的样本
    print("\n" + "=" * 60)
    print("Top 10 负面影响最大的训练样本")
    print("=" * 60)
    
    bottom_k = 10
    bottom_indices = np.argsort(influence_scores)[:bottom_k]
    
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"{rank:2d}. Sample {idx:6d}: score = {influence_scores[idx]:10.6f}")
    
    # 可视化
    print("\n生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Influence score 分布
    axes[0, 0].hist(influence_scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel("Influence Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Influence Score 分布")
    axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
    axes[0, 0].legend()
    
    # 2. Cumulative influence
    sorted_scores = np.sort(influence_scores)[::-1]
    cumsum = np.cumsum(sorted_scores)
    cumsum_normalized = cumsum / cumsum[-1] if cumsum[-1] != 0 else cumsum
    
    axes[0, 1].plot(cumsum_normalized)
    axes[0, 1].set_xlabel("样本数 (按 influence 排序)")
    axes[0, 1].set_ylabel("累积 Influence (归一化)")
    axes[0, 1].set_title("累积 Influence 曲线")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0.8, color='red', linestyle='--', label='80%')
    axes[0, 1].legend()
    
    # 3. Top vs Bottom
    top_100 = np.sort(influence_scores)[-100:]
    bottom_100 = np.sort(influence_scores)[:100]
    
    axes[1, 0].boxplot([top_100, bottom_100], labels=['Top 100', 'Bottom 100'])
    axes[1, 0].set_ylabel("Influence Score")
    axes[1, 0].set_title("Top vs Bottom 样本对比")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Percentile 分析
    percentiles = np.arange(0, 101, 5)
    percentile_values = np.percentile(influence_scores, percentiles)
    
    axes[1, 1].plot(percentiles, percentile_values, marker='o')
    axes[1, 1].set_xlabel("Percentile")
    axes[1, 1].set_ylabel("Influence Score")
    axes[1, 1].set_title("Percentile 分析")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    
    plt.tight_layout()
    
    output_path = pathlib.Path(result_path).parent / "influence_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {output_path}")
    
    return influence_scores


def example_select_top_data(
    result_path: str = "./results/influence.pkl",
    top_k: int = 1000,
    output_file: str = "./results/top_samples.txt",
):
    """示例: 选择 Top-K 数据用于 retraining."""
    
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    
    influence_scores = result["influence_scores"]
    
    # 选择 Top-K
    top_indices = np.argsort(influence_scores)[-top_k:][::-1]
    
    # 保存索引
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for idx in top_indices:
            f.write(f"{idx}\n")
    
    print(f"已保存 Top-{top_k} 样本索引到: {output_path}")
    
    # 统计
    top_scores = influence_scores[top_indices]
    print(f"\nTop-{top_k} 样本统计:")
    print(f"  Mean score: {top_scores.mean():.6f}")
    print(f"  Min score:  {top_scores.min():.6f}")
    print(f"  Max score:  {top_scores.max():.6f}")
    
    return top_indices


if __name__ == "__main__":
    print("OpenPI LoRA Influence 计算示例")
    print("=" * 60)
    
    # 示例 1: 收集 rollouts (注释掉,避免实际运行)
    # print("\n示例 1: 收集 eval rollouts")
    # example_collect_rollouts()
    
    # 示例 2: 运行 influence 计算 (注释掉,避免实际运行)
    # print("\n示例 2: 运行 influence 计算")
    # example_run_influence()
    
    # 示例 3: 分析结果 (需要先运行计算)
    print("\n示例 3: 分析 influence 结果")
    print("(注意: 需要先运行 influence 计算生成结果文件)")
    
    # example_analyze_results("./results/influence.pkl")
    
    # 示例 4: 选择 Top 数据
    print("\n示例 4: 选择 Top-K 数据")
    # example_select_top_data("./results/influence.pkl", top_k=1000)
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
