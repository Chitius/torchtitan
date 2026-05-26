本文档概述了在 `torchtitan` 仓库中添加新模型的流程。在大多数情况下，新模型应首先添加到 `torchtitan/experiments` 文件夹中。关于贡献标准，请参阅其中的[贡献指南](/torchtitan/experiments/README.md)。总体而言，请遵循 `torchtitan` 的[指导原则](/README.md#overview)。

对于离线探索，除非另有说明，我们推荐遵循相同的步骤。

## 添加模型

请参阅 [Llama 3 文件夹](llama3) 作为示例。

文件夹应按以下方式组织：

- `model.py`
  - 注意：请遵循指导原则，编写单设备模型代码。
  - 注意：我们优先保证可读性而非灵活性。首选风格是不在不同模型之间共享模块，除非是最常见和最复杂的模块。
  - 定义一个继承自基础模型（例如 `torchtitan/models/common/decoder.py` 中的 `Decoder`）的 Model 类。
  - 模型类应包含一个嵌套的 `Config` 数据类（继承自基础模型的 `Config`），用于保存所有架构超参数。
    - `get_nparams_and_flops()` 将用于了解模型大小和计算吞吐量。
    - `update_from_config()` 从训练配置更新模型配置（例如同步 seq_len、处理硬件特定设置）。
  - `__init__()` 接收 `Config` 来构建模型。
  - 参数初始化由每个模块 `Config` 上的 `param_init` 系统处理。在每个子配置的模型配置注册表中设置 `param_init`（一个 `dict[str, Callable]`，将参数名称映射到初始化函数）。`init_states()` 会自动递归到所有子模块，因此不需要手动递归调用。覆盖 `_init_self_buffers()` 以进行设备感知的缓冲区初始化（例如 RoPE、MoE）。
  - 如果 `model.py` 变得过大或过于复杂，可以添加额外的文件以降低其复杂度，例如用 `moe.py` 来存放 `MoE`、`Router` 和 `GroupedExperts` 模块。

- `state_dict_adapter.py`
  - 继承 [`BaseStateDictAdapter`](/torchtitan/protocols/state_dict_adapter.py)，以实现 `torchtitan` 模型定义与其他模型定义（例如来自 HuggingFace，以便我们可以以 HF 格式保存/加载模型检查点）之间的状态字典映射。
  - 此类适配器有多种使用方式：
    - `scripts/checkpoint_conversion/` 中的检查点转换脚本将使用它们来适配包含非分片 `torch.Tensor` 的状态字典（在 CPU 上）。
    - 在训练期间，[`CheckpointManager`](/torchtitan/components/checkpoint.py) 将使用它们来适配包含（可能已分片的）`DTensor` 的状态字典（在 GPU 上），以便以 HF 格式保存/加载检查点。
    - 在训练后，`to_hf()` 帮助将 torchtitan 模型转换为 HF 模型，可供其他框架用于推理。
  - 对于离线探索，这是可选的。

- `sharding.py`
  - 定义 `set_<model>_sharding_config(config, *, loss_parallel, enable_sp, ...)`，为模型配置中的每个 `Module.Config`（嵌入层、归一化层、注意力层、前馈层、输出层）填充 `sharding_config`。TP、SP 和内部注意力的 `LocalMapConfig` 放置通过 `ShardingConfig` 以声明式方式表达，而非运行时的 `parallelize_module` 计划。
  - 在 `Model.Config.update_from_config()` 中调用辅助函数，以便放置策略依赖于训练器的 `parallelism` 设置。
  - 尽可能重用 `torchtitan/models/common/decoder_sharding.py` 中的共享辅助函数（`set_decoder_sharding_config`、`set_dense_ffn_sharding`、`set_gqa_attention_sharding`、`norm_config`、`dense_param_placement`、`dense_activation_placement`）。
  - 在 `--training.full_dtensor` 下，按规范的外层到内层 SPMD 顺序声明网格轴：密集层（注意力/MLP/归一化/嵌入/lm_head）为 `(dp_replicate, dp_shard, cp, tp)`，稀疏层（MoE 专家权重）为 `(dp_replicate, efsdp, ep)`。`Module.parallelize` 按声明的顺序解析网格，并验证其是否与其中一个 SPMD 网格匹配；轴顺序声明错误将引发 `ValueError`。

- `parallelize.py`
  - 按以下顺序应用训练技术：
    - `model.parallelize(parallel_dims)` — 由 `sharding_config` 驱动的自动递归声明式分片（TP、SP、注意力 `local_map`）。替代每个模型的 `parallelize_module` 计划字典。
    - （MoE 模型）`apply_moe_ep_tp` 用于 MoE 专家上的专家并行 + TP（目前尚非基于配置）。
    - 激活检查点
    - `torch.compile`
    - FSDP / HSDP
    - 注意：目前语言模型的 CP 支持通过 `torchtitan/train.py` 中的上下文管理器启用。理想情况下，启用 CP 不需要额外的工作。

- `pipeline.py`（如果模型规模较小则可选）
  - 应用 PP

- `__init__.py`
  - 一个实际模型配置的字典，类型为 `[str: Model.Config]`。
  - 定义 `model_registry(flavor)` 以返回一个 [`ModelSpec`](/torchtitan/protocols/model_spec.py)，包含：
    - 模型名称和风格（flavor）
    - 模型配置（一个 `Model.Config` 数据类）
    - 并行化函数、流水线化函数
    - 损失函数构建器
    - 状态字典适配器
  - 模型名称应与文件夹名称相同，并添加到 `torchtitan/models/__init__.py` 或 `torchtitan/experiments/__init__.py`。
  - 阅读[更多](/docs/extension.md#modelspec)关于 `ModelSpec` 的信息。

- `config_registry.py`
  - 为每个训练配置定义一个函数（例如 `llama3_debugmodel`、`llama3_8b`、`llama3_70b`）。
  - 每个函数返回一个包含所有训练设置的 `Trainer.Config`（或子类）实例。
  - 函数可以通过变异相互派生以支持变体（例如 flex_attn、float8）。
  - 这些配置在运行时通过 `--module <model_name> --config <function_name>` 选择。

- `README.md`
  - 包含下载分词器/编码器的[说明](/README.md#downloading-a-tokenizer)。
  - 包含下载模型检查点以进行持续预训练或训练后处理的说明。
  - 更新开发状态，包括已支持的功能和即将推出的功能。
  - 对于离线探索，这是可选的。

## 测试与基准测试

- 数值测试
  - 一种端到端的方法是，将相同的模型检查点加载到 `torchtitan` 模型和 HF 模型中，并比较给定相同输入时模型的输出。这需要：
    - HF 实现是正确的。
    - `torchtitan` 模型的正确性以及相应的状态字典适配器的正确性，共同表明两者都是正确的。

- 损失收敛
  - 如果存在已验证的基线，请与基线比较损失曲线。
  - 对于 `torchtitan` 内部的比较，请参阅[指南](/docs/converging.md)。

- 性能基准测试
  - 请参阅 [benchmarks](/benchmarks/) 文件夹。

- CI 测试
  - 包括单元测试和集成测试，参见[示例](/tests/)。
  - 如果模型文件夹位于 experiments 文件夹下，请将测试放在模型文件夹下。否则，将测试放在 `/tests` 文件夹下。
  - 添加必要的 GitHub [工作流](/.github/workflows/)。
