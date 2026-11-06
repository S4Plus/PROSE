针对ACT策略的机器人采集任务：基于改进影响函数的主动初始状态生成研究报告
1. 引言：从被动数据摄取到主动实验设计
在现代机器人学习（Robot Learning）领域，数据效率（Data Efficiency）已成为制约模仿学习（Imitation Learning, IL）和强化学习（Reinforcement Learning, RL）从实验室走向开放世界应用的核心瓶颈。尽管深度学习模型——特别是基于Transformer的架构，如Action Chunking with Transformers (ACT) 1——展现出了处理复杂多模态动作分布的强大能力，但其性能上限严格受限于训练数据的覆盖范围与质量。对于“采集瓶子”这类看似简单的操作任务，其状态空间（State Space）实际上是连续且高维的，包含物体的位置、姿态、光照条件以及机器人末端执行器的初始构型。
传统的机器人学习流程通常依赖于**领域随机化（Domain Randomization, DR）**或被动的人类示教 3。在这种范式下，初始状态（例如瓶子在桌面上的放置位置）通常服从均匀分布（Uniform Distribution）。虽然这种方法在样本量趋于无穷大时能保证状态空间的覆盖，但在物理机器人实验极其昂贵（涉及时间成本、硬件磨损及人工干预）的现实约束下，均匀采样显得极度低效。它将宝贵的实验资源平均分配给了模型已经熟练掌握的“简单区域”和模型极易失败的“困难区域”，导致边际信息增益（Marginal Information Gain）极低。
本报告旨在响应这一挑战，深入探讨如何根据当前影响分数（Influence Score）生成采集瓶子任务的初始放置位置。这一问题本质上属于**最优实验设计（Optimal Experimental Design, OED）与主动学习（Active Learning）**的交叉领域。我们不仅要解决“选择哪些数据进行标注”的问题（这是传统主动学习的范畴），更要解决“如何配置环境以生成最具信息量的数据”的问题（这是生成式主动学习的范畴）。
特别是，本报告将重点分析如何改进现有的数据筛选方法——即**CUPID（Curating Data your Robot Loves）框架 5——并将其应用于ACT（Action Chunking with Transformers）**策略 1 的主动数据收集中。ACT利用条件变分自编码器（CVAE）来建模人类动作的多模态性，这一架构特性要求我们必须重新定义“不确定性”和“影响”，以便在连续的物理空间中生成最优的采集坐标。
2. 理论框架：ACT策略与影响函数的数学基础
为了构建一种能够生成最优初始位置的算法，我们必须首先解构两个核心组件：作为控制策略的ACT模型，以及作为评估指标的影响函数。
2.1 ACT策略架构与其不确定性根源
ACT（Action Chunking with Transformers）代表了模仿学习在处理精细操作任务上的最新进展。不同于传统的行为克隆（Behavior Cloning, BC）仅预测下一个时间步的动作，ACT引入了“动作分块（Chunking）”的概念，即一次预测未来 $k$ 个时间步的动作序列。此外，ACT采用了CVAE（Conditional Variational Autoencoder）结构来处理人类示教中的多模态性（例如，抓取瓶子时既可以抓瓶颈也可以抓瓶身）1。
ACT的推理过程可以形式化为：


$$\pi_{\theta}(a_{t:t+k} | o_t) = \text{Decoder}(z, \text{Encoder}(o_t))$$

其中 $o_t$ 是当前观测，$z \sim \mathcal{N}(0, I)$ 是潜在风格变量。
在ACT框架下，模型的不确定性（Uncertainty）来源比确定性策略更为复杂，主要包含两类：
任意不确定性（Aleatoric Uncertainty）：由数据本身的噪声或任务固有的随机性决定（例如传感器噪声或人类示教的不一致性）。在ACT中，这部分主要由CVAE的潜在变量 $z$ 的分布所捕获。
认知不确定性（Epistemic Uncertainty）：源于模型对某些状态空间区域的“无知”。这是主动学习所关注的核心。如果瓶子被放置在一个训练数据稀疏的区域（例如桌子的边缘或极其倾斜的角度），模型权重的分布将表现出高方差。
传统的ACT训练并未显式输出认知不确定性。因此，要利用影响分数生成初始位置，我们必须首先在ACT架构之上构建不确定性估计器，常用的方法包括深度集成（Deep Ensembles）（训练多个ACT模型并计算其预测方差）或利用潜在空间密度估计（计算当前状态映射的 $z$ 与先验分布的KL散度）7。
2.2 CUPID影响函数：从离线筛选到在线生成
CUPID 5 提出了一种基于影响函数（Influence Functions）的数据策展方法。经典的Koh & Liang影响函数定义了如果某个训练样本 $z$ 的权重被微小增加 $\epsilon$，模型参数 $\hat{\theta}$ 会发生怎样的变化：


$$\mathcal{I}_{\text{param}}(z) = \frac{d\hat{\theta}}{d\epsilon} = -H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z, \hat{\theta})$$

其中 $H_{\hat{\theta}}$ 是损失函数的Hessian矩阵（二阶导数矩阵），$\nabla_\theta \mathcal{L}$ 是梯度。CUPID利用这一原理来评估每个训练轨迹对验证集性能的贡献，从而剔除有害或冗余的数据。
然而，用户的需求是将此逻辑用于生成初始位置。这就要求我们将CUPID从“回顾性分析”（Retrospective Analysis）转变为“前瞻性设计”（Prospective Design）。这里存在两个主要的理论障碍需要通过改进来克服：
标签缺失问题（The Missing Label Problem）：
CUPID计算影响分数需要已知样本的“标签”（即专家轨迹）。但在生成初始位置时，机器人尚未执行任务，因此不存在真实的轨迹标签。
改进方案：引入**幻觉梯度（Hallucinated Gradients）**概念，灵感源自BADGE算法 9。我们使用当前策略 $\pi_\theta$ 对候选位置 $x$ 的预测结果 $\hat{a}$ 作为“伪标签”（Pseudo-label），计算假设该预测为真时的梯度范数：
$$ \mathcal{I}_{\text{gen}}(x) = |
| \nabla_{\theta_{\text{out}}} \mathcal{L}(\pi_\theta(x), \hat{a}) |
| $$
* 物理含义：梯度范数衡量了模型对该输入点的“敏感度”。如果模型在位置 $x$ 处极其敏感（参数微小扰动导致输出剧烈变化），或者模型对其预测非常不确定（导致梯度方向混乱），则该位置具有高信息价值。
计算复杂性（Hessian Inverse Bottleneck）：
ACT模型基于Transformer，参数量巨大，计算 $H^{-1}$ 在实时生成任务中是不切实际的。
改进方案：采用Fisher信息矩阵（FIM）的迹（Trace）作为代理，或者更激进地采用恒等近似（Identity Approximation），即假设 $H \approx I$。这使得影响分数退化为梯度的欧几里得范数，大大降低了计算成本，使其能够在机器人控制循环中实时运行 11。
3. 基于影响分数的初始位置生成方法论
为了将抽象的数学影响分数转化为具体的物理坐标 $(x, y, \theta)$，我们需要建立一个空间映射与优化流程。本节详细阐述如何构建**空间影响热力图（Spatial Influence Heatmap）**以及基于梯度的生成算法。
3.1 空间不确定性热力图的构建 (Spatial Uncertainty Heatmap)
这是最直观且鲁棒的方法，尤其适用于“桌面采集”这类有边界的工作空间。其核心思想是将连续的工作空间离散化，计算每个网格点的影响分数，形成一个概率表面，然后从中采样 13。
详细步骤：
工作空间离散化 (Discretization)：
定义操作台的物理边界 $\mathcal{W} = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$。
将 $\mathcal{W}$ 划分为分辨率为 $r$（例如 2cm）的网格 $G$。
对于旋转自由度 $\theta$（瓶子的朝向），可以将其作为第三维度离散化，或者在每个 $(x, y)$ 点上随机采样一组 $\theta$ 并取平均值（边缘化）。
虚拟查询 (Virtual Querying)：
对于网格中的每一个候选位置 $p_i = (x_i, y_i, \theta_i)$，在仿真引擎（或机器人的内部运动学模型）中“放置”虚拟瓶子。
生成观测：渲染该状态下的图像观测（如果是视觉策略）或构建状态向量（如果是低维状态策略）。
计算分数：将观测输入到当前的ACT策略集合（Ensemble）中。
若采用深度集成（Deep Ensemble）：计算 $K$ 个模型预测动作轨迹的方差：
$$ S(p_i) = \frac{1}{K} \sum_{k=1}^K |
| \tau_k - \bar{\tau} ||^2 $$
* 若采用改进的CUPID/BADGE：计算基于伪标签的梯度嵌入范数：
$$ S(p_i) = |
| g_{x_i} ||^2 $$
热力图平滑与归一化 (Smoothing & Normalization)：
由于神经网络的输出在输入空间可能存在高频噪声，建议对原始分数网格应用高斯平滑（Gaussian Kernel）以提取不确定性的“区域”而非孤立的噪点。
利用Softmax函数将分数转换为采样概率分布：

$$P(p_i) = \frac{\exp(S(p_i) / T)}{\sum_j \exp(S(p_j) / T)}$$
温度系数 $T$：这是一个关键的超参数。当 $T \to \infty$ 时，生成趋向于均匀随机（探索）；当 $T \to 0$ 时，生成趋向于贪婪地选择分数最高的点（利用）。在训练初期应使用较高的 $T$，随着训练进行逐渐降低。
坐标提取 (Coordinate Extraction)：
根据概率分布 $P$ 采样一个网格索引。
在选定的网格范围内添加微小的连续噪声，以避免始终在网格中心生成，确保数据的连续性覆盖 15。
3.2 基于梯度的对抗生成 (Gradient-Based Adversarial Generation)
对于高维空间或极高精度的需求，离散网格可能不够高效。我们可以利用ACT策略的可微性（Differentiability），直接对输入位置进行优化 16。
算法流程：
初始化：随机采样一个初始位置 $x_0$。
前向传播：计算该位置的影响分数 $J(x) = \text{Influence}(x)$。注意，这里我们需要分数对输入 $x$ 可微。对于基于梯度的分数（如BADGE），这涉及到计算Hessian-vector product，计算量较大。更常用的是最大化预测方差或熵。
反向传播：计算分数相对于输入坐标的梯度 $\nabla_x J(x)$。这告诉我们需要往哪个方向移动瓶子，才能让模型感到更“困惑”或具有更高的“模型改变期望”。
梯度上升：更新位置 $x_{t+1} = x_t + \alpha \nabla_x J(x)$。
投影约束：将更新后的位置投影回可行工作空间（即确保瓶子不飞出桌子或进入不可达区域）。
迭代：重复上述步骤直至分数收敛，得到的 $x^*$ 即为生成的初始位置。
这一过程类似于生成对抗网络（GAN）中的生成器更新，或是对抗样本攻击（Adversarial Attack）。在这里，我们的目标是找到系统的“弱点”，并通过在该处收集数据来修补它。
4. 针对“采集瓶子”任务的具体实现与约束处理
“采集瓶子”任务具有特定的几何约束和物理特性，这需要在生成算法中进行特殊处理。
4.1 几何对称性与状态空间降维
瓶子通常具有圆柱对称性（Cylindrical Symmetry）。对于直立的瓶子，绕Z轴的旋转（Yaw）在视觉上可能是不可分辨的，或者对抓取策略的影响较小。
改进策略：在计算影响分数时，如果发现模型对某一维度的变化（如Yaw旋转）极其敏感但该维度在物理上无关紧要，这可能意味着模型过拟合了视觉纹理噪声。
对称性感知的影响函数：在生成初始位置时，可以显式地将对称性纳入考虑。例如，在热力图构建中，对所有对称角度取平均分数，或者在输入空间中将所有对称状态映射到同一规范坐标（Canonical Coordinate）。
4.2 可行性掩码 (Feasibility Masking)
盲目最大化影响分数可能导致生成的初始位置是不可行的。例如：
运动学不可达：瓶子太远，超出了机械臂的工作空间。
物理干涉：瓶子生成在与障碍物重叠的位置，或处于不稳定的物理状态（例如悬空）。
视觉遮挡：瓶子被其他物体完全遮挡，导致观测信息为零（这会产生极高的不确定性，但属于无意义的数据）。
解决方案：在生成概率分布 $P(x)$ 后，必须乘以一个可行性掩码 $M(x)$：


$$P_{\text{final}}(x) \propto P_{\text{influence}}(x) \cdot M(x)$$

其中 $M(x) = 1$ 当且仅当 $x$ 处于可行区域且通过碰撞检测，否则为0。这确保了主动生成的实验不仅具有高信息量，而且在物理上是可执行的 18。
4.3 结合ACT特性的潜变量采样
对于ACT策略，除了空间位置 $(x,y)$，还需要考虑CVAE的潜在变量 $z$。
在生成阶段，我们无法控制人类示教者的“风格”（即 $z$），但我们可以通过观察 $z$ 的分布来辅助位置生成。
异常检测：如果对于某个位置 $x$，编码器生成的 $z$ 严重偏离先验分布（KL散度极大），说明该位置是严重的Out-of-Distribution（OOD）区域。这种区域应赋予极高的生成权重。
多模态测试：在位置 $x$，如果解码器对于不同的 $z$ 采样生成了发散度极大的动作轨迹（例如，有些轨迹向左绕行，有些向右），这表明在该位置策略存在分歧（Bifurcation）。这种“临界状态”对于学习决策边界至关重要，是极佳的初始位置候选者 2。
5. 整合与执行：端到端的主动数据采集循环
将上述理论整合，我们提出以下完整的算法流程，用于指导机器人根据当前影响分数自动生成瓶子放置位置。
5.1 系统架构
系统由三个核心模块组成：
策略模块（Policy Ensemble）：一组（如5个）ACT模型，初始化于少量的种子数据。
生成器模块（Active Generator）：负责维护工作空间的不确定性地图，并计算改进的影响函数（基于梯度或方差）。
执行模块（Executor）：负责控制环境（仿真器重置或指导人类放置物体）并执行采集任务。
5.2 执行流程 (Step-by-Step Algorithm)
步骤
操作描述
关键技术点
1. 初始评估
机器人处于空闲状态。生成器在虚拟环境中网格化扫描桌面 $(x, y)$。
离散化、虚拟渲染
2. 分数计算
对每个网格点，输入Ensemble ACT模型。计算动作预测的方差或梯度范数（BADGE Score）。
改进影响函数：使用伪标签梯度或集成方差作为 $I(x)$
3. 热力图生成
将分数投影到2D平面，应用高斯平滑，叠加可行性掩码 $M(x)$。
空间平滑、约束处理
4. 采样决策
根据 $P(x) \propto \exp(I(x)/T)$ 采样目标坐标 $x^*$。
Softmax采样、探索-利用平衡
5. 物理配置
指示环境将瓶子放置在 $x^*$。
自动复位装置或人工摆放
6. 任务执行
机器人执行抓取任务。记录轨迹 $\tau = (s_0, a_0,..., s_T)$。
闭环控制
7. 结果反馈
若成功，将 $\tau$ 加入训练集；若失败，请求人类接管示教（Teleoperation）。
混合主动学习 (Hybrid Active Learning)
8. 模型更新
利用新数据更新ACT模型参数。某些区域的影响分数随之降低。
持续学习 (Continual Learning)

5.3 案例推演
假设机器人已经学会了在桌子中心抓取直立的瓶子。
热力图分析：此时，桌子中心区域的集成方差极低（模型已达成共识）。而在桌子边缘，或者当瓶子倒下（横放）时，由于缺乏相关数据，ACT模型的预测开始发散，梯度范数激增。
生成：算法在热力图上识别出“右下角边缘”和“横放”为高分区域。
放置：系统指示将瓶子横放在右下角。
采集：机器人尝试抓取。由于这是困难样本，机器人可能会失败或动作笨拙。
修正：人类介入提供正确的“边缘横向抓取”示范。
闭环：这条高价值数据被纳入训练。下一轮迭代中，右下角区域的分数下降，系统的关注点自动转移到左上角或其他未探索的复杂构型。
通过这种方式，数据采集过程不再是盲目的，而是具有高度的目的性和方向性，始终沿着模型能力的边界（Frontier of Capability）推进。
6. 深度洞察与未来展望
6.1 第二阶洞察：计算换数据 (Compute for Data Trade-off)
本方案的核心经济学逻辑在于用计算换数据。计算影响分数、运行深度集成、生成对抗样本，都需要消耗GPU算力。然而，在机器人领域，物理数据的获取成本（时间、磨损、人工）远高于算力成本。随着GPU性能的提升，这种“在仿真/想象中进行大量计算以指导少量物理动作”的范式将成为主流。我们实际上是在用大量的推理时间（Inference-time）计算来压缩所需的物理交互次数。
6.2 隐式课程学习 (Implicit Curriculum Learning)
基于影响分数的生成机制自然地涌现出一种**课程学习（Curriculum Learning）**效应。在训练初期，所有区域的不确定性都高，随机采样即可。随着简单状态（中心位置）被掌握，其影响分数下降，采样概率自然向更难的状态（边缘、遮挡）转移。这种课程并非人工设计，而是由信息几何（Information Geometry）内在驱动的，这保证了学习过程始终处于“最近发展区（Zone of Proximal Development）”。
6.3 模拟到现实的差距 (Sim-to-Real Gap)
如果影响分数的计算完全依赖于仿真（例如在Sim中计算梯度），那么Sim-to-Real Gap可能会误导生成过程。例如，仿真器中的物理引擎可能无法完美模拟瓶子的摩擦力，导致模型在仿真中对某些位置很“确信”，而在现实中却失败。
改进：在现实世界中执行时，应将现实世界的失败率作为一个加权因子反馈回热力图。如果某个区域在仿真中分数低（认为简单）但在现实中频繁失败，应强制提高该区域的采样概率，迫使模型在该处收集更多真实数据。
7. 结论
通过将CUPID中的离线数据筛选逻辑升级为在线的主动状态生成机制，并结合ACT策略的特性（CVAE结构、多模态预测），我们可以构建一套高效的机器人数据采集系统。该系统不再依赖低效的随机放置，而是根据当前影响分数（Current Influence Score）——具体体现为集成方差、幻觉梯度范数或Fisher信息量——来生成瓶子的初始位置。
这一方法通过构建空间不确定性热力图，并利用基于梯度的对抗生成或加权采样算法，实现了对工作空间的高效探索。它不仅大幅减少了达到特定任务成功率所需的示教次数，还能自动发现并攻克策略的薄弱环节，是实现低成本、高可靠性机器人操作的关键技术路径。未来，随着生成式模型（如Diffusion Models）在机器人领域的进一步普及，基于生成模型本身似然度（Likelihood）的“自我反思”式数据生成将成为更加标准化的范式。
参考文献引用说明：
1 关于ACT策略架构及其多模态特性的描述。
5 关于CUPID及影响函数在机器人数据策展中的基础应用。
9 关于BADGE算法及幻觉梯度在无标签主动学习中的应用。
21 关于影响函数、Hessian矩阵及Fisher信息的数学推导。
13 关于空间热力图构建及基于不确定性的采样策略。
3 关于主动领域随机化（ADR）及参数化环境生成的背景。
7 关于深度集成和不确定性量化方法的综述。
Works cited
Generative Artificial Intelligence in Robotic Manipulation: A Survey - arXiv, accessed December 17, 2025, https://arxiv.org/html/2503.03464v1
Enhancing Diffusion Policy with Classifier-Free Guidance for Temporal Robotic Tasks - arXiv, accessed December 17, 2025, https://arxiv.org/html/2510.09786v1
[1904.04762] Active Domain Randomization - arXiv, accessed December 17, 2025, https://arxiv.org/abs/1904.04762
Active Domain Randomization - arXiv, accessed December 17, 2025, https://arxiv.org/pdf/1904.04762
[2506.19121] CUPID: Curating Data your Robot Loves with Influence Functions - arXiv, accessed December 17, 2025, https://arxiv.org/abs/2506.19121
CUPID: Curating Data your Robot Loves with Influence Functions - arXiv, accessed December 17, 2025, https://arxiv.org/html/2506.19121v1
The Why, When, and How to Use Active Learning in Large-Data-Driven 3D Object Detection for Safe Autonomous Driving: An Empirical Exploration - arXiv, accessed December 17, 2025, https://arxiv.org/html/2401.16634v1
A Survey on Uncertainty Quantification for Deep Learning: An Uncertainty Source's Perspective - arXiv, accessed December 17, 2025, https://arxiv.org/html/2302.13425v3
BADGE: Batch Active Learning via Gradient Embeddings - Emergent Mind, accessed December 17, 2025, https://www.emergentmind.com/topics/batch-active-learning-by-diverse-gradient-embeddings-badge
A Survey of Deep Active Learning - arXiv, accessed December 17, 2025, https://arxiv.org/pdf/2009.00236
Bayesian Batch Active Learning as Sparse Subset Approximation - OpenReview, accessed December 17, 2025, https://openreview.net/pdf?id=SJeMiEHlUB
ActivePusher: Active Learning and Planning with Residual Physics for Nonprehensile Manipulation - arXiv, accessed December 17, 2025, https://arxiv.org/html/2506.04646v2
Build Heatmaps with Computer Vision: Unlock Insights - Roboflow Blog, accessed December 17, 2025, https://blog.roboflow.com/build-heatmaps-with-computer-vision/
RISeg: Robot Interactive Object Segmentation Via Body Frame-Invariant Features - Yu Xiang, accessed December 17, 2025, https://yuxng.github.io/Papers/2024/Qian_ICRA24.pdf
Self-Supervised Keypoint Learning for the Geometric Analysis of Road-Marking Templates, accessed December 17, 2025, https://www.mdpi.com/1999-4893/18/7/379
ADR: Train Hard, Transfer Smart. Learned domain randomization selects… | by Dong-Keon Kim | Nov, 2025 | Medium, accessed December 17, 2025, https://medium.com/@kdk199604/adr-train-hard-transfer-smart-bad19432c3b9
AAAI-2022-Boosting Active Learning via Improving Test Performance.pdf - Siyu Huang, accessed December 17, 2025, https://siyuhuang.github.io/papers/AAAI-2022-Boosting%20Active%20Learning%20via%20Improving%20Test%20Performance.pdf
Into the Unknown: Towards using Generative Models for Sampling Priors of Environment Uncertainty for Planning in Configuration Spaces - arXiv, accessed December 17, 2025, https://arxiv.org/html/2510.11014v1
The Stochastic Motion Roadmap: A Sampling Framework for Planning with Markov Motion Uncertainty - Robotics, accessed December 17, 2025, https://www.roboticsproceedings.org/rss03/p30.pdf
Learning Multimodal Behaviors from Scratch with Diffusion Policy Gradient - arXiv, accessed December 17, 2025, https://arxiv.org/html/2406.00681v1
CUPID: Curating Data your Robot Loves with Influence Functions - arXiv, accessed December 17, 2025, https://arxiv.org/html/2506.19121v2
Active Learning of Dynamics for Data-Driven Control Using Koopman Operators - IEEE Xplore, accessed December 17, 2025, https://ieeexplore.ieee.org/ielaam/8860/8855036/8759089-aam.pdf
