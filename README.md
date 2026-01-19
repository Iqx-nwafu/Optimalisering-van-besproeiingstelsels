该仓库用强化学习和其他优化方法解决灌溉系统分组问题。代码分为若干模块：水力模拟器、强化学习环境与训练脚本、基线方法以及参数调优脚本。理解各脚本的功能和依赖可以帮助合理安排运行顺序。

1. 数据准备与水力模拟器

输入文件： 项目依赖两份 Excel 表格 Nodes.xlsx 和 Pipes.xlsx，分别包含节点标高和管道长度/直径/材质等信息。这些文件需放在项目根目录。

水力模拟器（tree_evaluator.py）：

定义 TreeHydraulicEvaluator 类，构建树状管网并实现快速稳态水力计算。

支持根据给定分组(open_laterals)计算每个节点的压力头和最小裕度，用于评价分组是否满足压力约束。

函数 load_nodes_xlsx 和 load_pipes_xlsx 用于读取 Excel 文件并生成节点和管道对象。

需要根据国家标准在 GBT_COEFS 中填入水头损失系数。

**运行顺序：**先准备好 Nodes.xlsx 和 Pipes.xlsx，若需要修改水头损失系数，则先在 tree_evaluator.py 中完善 GBT_COEFS。

2. 强化学习环境与配置

环境定义（ppo_env.py）：

实现 IrrigationGroupingEnv，继承自 Gymnasium 的 Env，为 MaskablePPO 创建 120 步的决策序列。

提供 action_masks() 以屏蔽已选择的支管；在 step 中根据选中4条支管后调用水力评估器并返回奖励。

环境构造函数接受 RewardWeights、RewardScales、SafetyShaping 等对象，用于定义奖励函数各项权重。

比较配置（comparison_utils.py）：

定义 ComparisonConfig 数据类，集中管理环境共有的默认参数。

提供 build_environment(seed, config) 创建统一的环境实例。

函数 evaluate_order 用于在环境中执行固定顺序，返回末次方差、最小裕度等指标。

定义 composite_score，将方差和最小裕度组合成单一评分。

**运行顺序：**一般无需手动调用环境模块，只需保证相关脚本能找到这些模块即可。

3. Maskable PPO 强化学习训练

训练脚本（train_maskable_ppo.py）：

使用 multiprocessing 创建并行环境（默认 num_envs=6），通过 _load_base() 读取 Excel 数据并缓存节点/管道及其相关信息。

函数 make_vec_env 根据 ComparisonConfig 创建多环境实例；make_eval_vec_env 创建单环境用于评估。

主函数中设置环境数量、Torch 线程数、构造训练和评估环境，然后创建 MaskablePPO 模型并调用 model.learn 训练 2,000,000 步。

训练结束后将模型保存为 ppo_irrigation_masked。

建议运行顺序：

确保安装依赖（stable-baselines3、sb3-contrib、gymnasium、pandas 等）。

运行 train_maskable_ppo.py 进行训练，可通过环境变量 N_ENVS、TORCH_THREADS 控制并行环境数量和线程数。

训练完成后会在当前目录产生 ppo_irrigation_masked.zip（sb3 默认格式）以及最优模型保存在 ./best_model/。

4. 强化学习模型评估与采样

评估脚本（eval_one.py）：

从 train_maskable_ppo.py 导入 make_env 并创建评估环境。

加载训练好的模型 ppo_irrigation_masked，先用确定性策略跑一条基线并输出方差、平均裕度等。

然后循环采样 N_SAMPLES 次（默认 5000 次），使用随机策略生成行动序列，筛选可行的样本并维护最小方差的前 TOPK 条记录。

最终将最优解和 Top‑k 结果保存为 JSON 文件，并在控制台输出结果概览。

**运行顺序：**在训练完成后运行 eval_one.py。可调整采样次数 (N_SAMPLES) 和保存的 Top‑k 值 (TOPK)。生成的 JSON 日志位于 ./sample_logs/ 目录。

5. 其他优化方法

简单神经网络基线（dnn_solution.py）：

使用 comparison_utils.build_environment 创建环境，并随机采样多个完整顺序以生成训练数据。

SimpleNN 是一个两层神经网络，用于根据静态特征预测每个支管的优先级。

训练完成后，按网络的预测顺序排序支管，调用 evaluate_permutation 计算最后方差和最小裕度。

NSGA‑II 多目标优化（nsga2_solution.py）：

定义标准 NSGA‑II 算法，包括非支配排序、拥挤距离和遗传算子（交叉和变异）。

通过 build_environment 创建环境，初始随机种群为所有支管的排列。

迭代若干代后输出最终群体的帕累托前沿及综合评分最优解。

奖励权重搜索（tune_reward_weights_grid.py）：

枚举不同的奖励权重组合，针对每组权重训练 MaskablePPO 一段时间并在验证环境中评估。

计算综合得分（方差 + 安全惩罚），记录所有配置的结果并找出最佳权重。

**运行顺序：**这些脚本可独立运行，用于探索不同算法或奖励配置。由于训练和评估耗时较长，建议在主要训练与评估流程之后，再根据需要运行。
