# torchtitan 模型定义完全指南

> 本文档在官方 `README.md` 的基础上做了大幅展开，目的是让读者在动手写一个**新模型**时，对"需要创建哪些文件、每个文件干什么、它们之间如何配合"有一个端到端的清晰认识。

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [按文件拆解：手把手教你写每个文件](#2-按文件拆解手把手教你写每个文件)
3. [最小可行模型与并行扩展路径](#3-最小可行模型与并行扩展路径)
4. [parallel.py 与 sharding.py 的配合关系](#4-parallelpy-与-shardingpy-的配合关系)
5. [Layer Config 组装时的维度契约与检查机制](#5-layer-config-组装时的维度契约与检查机制)
6. [Tokenizer 复用与无 HF 仓模型](#6-tokenizer-复用与无-hf-仓模型)
7. [完整示例：从 Llama3 出发做减法](#7-完整示例从-llama3-出发做减法)

---

## 1. 系统架构总览

torchtitan 的模型系统可以抽象为三层：

```
┌─────────────────────────────────────────────────────────────┐
│  协议层 (protocols/)                                         │
│  BaseModel, Module, ModelSpec, ShardingConfig,               │
│  StateDictAdapter ...                                        │
├─────────────────────────────────────────────────────────────┤
│  共享组件层 (models/common/)                                 │
│  Decoder, TransformerBlock, GQAttention, FeedForward,        │
│  MoE, Linear, Embedding, RMSNorm, RoPE ...                   │
├─────────────────────────────────────────────────────────────┤
│  模型特化层 (models/<your_model>/)                           │
│  model.py, __init__.py, parallelize.py, sharding.py,         │
│  state_dict_adapter.py, config_registry.py                   │
└─────────────────────────────────────────────────────────────┘
```

**核心设计原则**：
- **单设备优先**：`model.py` 里只写单卡逻辑，不写任何 `torch.distributed` 代码。
- **配置即代码**：所有子模块通过嵌套的 `Config` dataclass 描述，训练时调用 `config.build()` 生成模块实例。
- **声明式并行**：TP/SP/EP 的切分策略通过 `ShardingConfig` "声明"在 Config 上，由通用引擎 `Module.parallelize()` 统一执行。
- **数据流隐式检查**：层与层之间的维度匹配主要依赖 PyTorch 运行时 shape 检查，Config 组装阶段只保留"可提前发现"的结构性检查（如 TP 可除性）。

---

## 2. 按文件拆解：手把手教你写每个文件

### 2.1 `model.py` —— 定义网络结构（单设备）

这是唯一一个真正描述"模型长什么样"的文件。`model.py` 里**严禁**出现 `torch.distributed` 相关代码。

#### 2.1.1 基类选择：`Decoder` 还是 `BaseModel`？

| 基类 | 适用场景 | 自带功能 |
|------|----------|----------|
| `Decoder` | 自回归 decoder-only 语言模型（Llama/Qwen/DeepSeek 等） | `__init__` 自动构建 tok_embeddings → layers → norm → lm_head；`forward` 已实现；`init_states` 已支持；`get_attention_masks` 已支持 flex/varlen dispatch |
| `BaseModel` | 非自回归模型（如扩散模型 Flux）或特殊架构 | 只有 `init_states` 递归和 `verify_module_protocol`；`forward` 和结构完全自定义 |

**99% 的情况下你都应该继承 `Decoder`**。继承后你的 `model.py` 通常只需要写两层：
1. 一个 `TransformerBlock` 子类（描述一层怎么拼）
2. 一个 `Decoder` 子类（描述全局 Config 和特殊逻辑）

#### 2.1.2 `TransformerBlock` 子类必须怎么写

```python
from dataclasses import dataclass
import torch
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.attention import AttentionMasksType

class MyTransformerBlock(TransformerBlock):
    """
    你的 Transformer 层。
    """

    # 1. 必须声明嵌套 Config，即使空继承也要写
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # 如果不需要新增字段，直接 pass
        pass

    # 2. __init__ 必须接收 config，并显式构建每个子模块
    def __init__(self, config: Config):
        super().__init__()
        # attention 和 feed_forward/moe 至少有一个非 None
        self.attention = config.attention.build()
        if config.feed_forward is not None:
            self.feed_forward = config.feed_forward.build()
        if config.moe is not None:
            self.feed_forward = config.moe.build()  # 命名可自定义，但 Decoder.forward 里不会访问它
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    # 3. forward 的签名必须与 Decoder.forward 中的调用严格一致
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

**为什么不能省略 `class Config` 的空声明？**

因为 `TransformerBlock` 自身的 `Config` 使用了 `@dataclass(kw_only=True, slots=True)`。Python dataclass 的继承链要求每一层都要重新声明 dataclass 装饰器，否则子类的 Config 不会生成正确的 `__init__`。`pass` 表示"我没有新增字段，但我要继承上一层的 dataclass 语义"。

**`forward` 签名的四个参数分别是什么？**

| 参数 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `x` | `torch.Tensor` | 上一层输出 / `tok_embeddings(tokens)` | shape `[batch_size, seq_len, dim]` |
| `freqs_cis` | `torch.Tensor` | `Decoder` 的 `self.freqs_cis` buffer | RoPE 缓存，会被所有层共享 |
| `attention_masks` | `AttentionMasksType \| None` | `Decoder.get_attention_masks()` | FlexAttention 用 `BlockMask`，Varlen 用 `VarlenMetadata`，SDPA 用 `None` |
| `positions` | `torch.Tensor \| None` | dataloader | 每个 token 的绝对位置，用于文档边界检测（`block_causal`）和 iRoPE（Llama4） |

如果你的模型不需要 RoPE（如使用绝对位置编码），你仍然要保留 `freqs_cis` 参数，但可以在 `forward` 里忽略它。这是接口契约，不能改签名。

#### 2.1.3 `Decoder` 子类必须怎么写

```python
from dataclasses import dataclass
import torch
import torch.nn as nn
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.utils import get_dense_model_nparams_and_flops

class MyModel(Decoder):
    """
    你的顶层模型。
    """

    # 1. 嵌套 Config 继承 Decoder.Config
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096                # 模型隐藏维度
        vocab_size: int = 128256       # 词表大小
        enable_weight_tying: bool = False

        # 2. 必须实现 update_from_config
        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            """
            在模型构建之前被 Trainer 调用。
            用途：
            - 同步 training.seq_len 到 rope.max_seq_len
            - 检查并行度数与模型维度的兼容性
            - 调用 set_sharding_config 填充 sharding_config（如果要做 TP）
            """
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len

            # 同步 rope 的 max_seq_len
            import dataclasses
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            # 检查 TP 兼容性
            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(f"tp ({tp}) must divide n_heads ({n_heads})")
                if n_kv_heads % tp != 0:
                    raise ValueError(f"tp ({tp}) must divide n_kv_heads ({n_kv_heads})")

            # 如果定义了 sharding.py，在这里调用
            # from .sharding import set_my_model_sharding_config
            # set_my_model_sharding_config(self, loss_parallel=..., enable_sp=...)

        # 3. 必须实现 get_nparams_and_flops
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """
            返回 (参数数量, FLOPs)。
            可以直接复用 torchtitan 提供的 helper。
            """
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    # 4. __init__ 通常很薄
    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    # 5. 可选：重写 init_states 处理 weight tying 的时序问题
    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            # 在参数初始化前重新绑定，确保两个模块指向同一参数
            self.tok_embeddings.weight = self.lm_head.weight
        super().init_states(buffer_device=buffer_device)
```

**`update_from_config` 的核心注意事项：**

- 它是在 **Config 对象上** 被调用的，此时 `nn.Module` 还没被构建。所以你不能访问 `self.layers`（这是 Module 的属性），但可以通过 `self.layers[0].attention.n_heads` 访问 Config 树。
- 如果你要修改 `self.rope`，注意 `slots=True` 的 dataclass 不支持直接赋值，必须用 `dataclasses.replace`。
- 所有可能让训练直接崩溃的结构性错误（如 TP 不可除、PP+weight tying 冲突）都应该在这里抛出，而不是等到模型构建后。

**`get_nparams_and_flops` 为什么必须实现？**

Trainer 会在日志中打印模型参数数量和理论 FLOPs，用于估算 throughput。如果你不想自己算，直接用 `get_dense_model_nparams_and_flops`（dense 模型）或 `get_moe_model_nparams_and_flops`（MoE 模型）即可。

#### 2.1.4 `init_states` 与 `param_init` 系统详解

torchtitan 不依赖 PyTorch 默认的 `reset_parameters()`，而是使用一套显式的 `param_init` 系统：

1. 每个 `Module.Config` 有一个可选字段 `param_init: dict[str, Callable]`。
2. `config.build()` 时，这个 dict 会被挂到 Module 实例的 `_param_init` 属性上。
3. `model.init_states()` 会递归调用所有子模块的 `init_states()`，最终调用 `_init_self_parameters()`。
4. `_init_self_parameters()` 遍历 `self.named_parameters(recurse=False)`，对每个 param 在 `_param_init` 中查找对应的初始化函数并执行。

**这意味着：你的每一个参数都必须有一个 initializer，否则会报错。**

常见写法：

```python
from functools import partial
import torch.nn as nn
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init

# 全局常量：所有 Linear 的 weight/bias 都这么初始化
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}

# Norm 只有一个 weight
_NORM_INIT = {"weight": nn.init.ones_}

# Embedding 初始化
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}

# 如果做 weight tying，tok_embeddings 的初始化要跳过（因为 lm_head 会初始化它）
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}

# 层号相关的 depth-scaled init
# depth_scaled_std(base_std, layer_id) = base_std / sqrt(2 * (layer_id + 1))
def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }

# lm_head 有时需要特殊 std
def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim ** -0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3*s, b=3*s),
        "bias": nn.init.zeros_,
    }
```

然后在 builder 中挂载：

```python
attention = make_gqa_config(
    dim=dim, n_heads=n_heads,
    wqkv_param_init=_LINEAR_INIT,
    wo_param_init=_depth_init(layer_id),  # 每层不同
    inner_attention=inner_attention,
    mask_type=mask_type,
)
```

**陷阱**：如果你自定义了一个新层（比如新的位置编码模块），它的参数名必须都在 `param_init` 中有对应条目。如果漏了，会在 `init_states()` 时抛出：

```
ValueError: No initializer for parameter 'foo' in MyModule.
```

**Buffer 初始化**：如果模块有 buffer（如 RoPE 的 `cache`、MoE 的 `expert_bias`），不要放在 `param_init` 中。而是重写 `_init_self_buffers(self, *, buffer_device)` 方法，在里面用 `torch.device(buffer_device)` 创建 buffer。

#### 2.1.5 Pipeline Parallel 兼容的 passthrough 设计

`Decoder.forward` 中有以下 passthrough 逻辑：

```python
h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
...
h = self.norm(h) if self.norm is not None else h
...
output = self.lm_head(h) if self.lm_head is not None else h
```

这是为 PP 准备的。`pipeline_llm` 在切分 stage 时，会把不属于本 stage 的模块 **prune 为 None**。例如：
- 第 0 个 stage 只包含 `tok_embeddings` + 前几个 block，那么 `norm` 和 `lm_head` 会被设为 `None`。
- 最后一个 stage 只包含后几个 block + `norm` + `lm_head`，那么 `tok_embeddings` 会被设为 `None`。

**你不需要在 `model.py` 里写任何 PP 特殊逻辑**，只要确保你的 `TransformerBlock.forward` 签名兼容，且 `Decoder` 基类的 passthrough 足够用即可。

---

### 2.2 `__init__.py` —— 组装配置并注册到训练框架

这个文件是模型与 torchtitan Trainer 之间的**胶水层**。它的核心任务是：**把一堆超参和初始化函数，组装成一个完整的 Model.Config 树，然后注册到 ModelSpec。**

#### 2.2.1 文件结构总览

一个典型的 `__init__.py` 包含以下部分（按出现顺序）：

1. **import**
2. **全局 init 常量**（`_LINEAR_INIT`, `_NORM_INIT`, `_EMBEDDING_INIT` 等）
3. **辅助 builder 函数**（`_build_layers`, `_make_attention_config` 等）
4. **多个模型尺寸 builder**（`_debugmodel`, `_8b`, `_70b` 等），每个返回一个填好的 `Model.Config`
5. **config 字典**（`my_model_configs = {"debugmodel": _debugmodel, ...}`）
6. **`model_registry(flavor, ...)` 函数**，返回 `ModelSpec`

#### 2.2.2 `_build_layers`：不要手写每层配置

常见错误是手写 `layers = [TransformerBlock.Config(...), TransformerBlock.Config(...), ...]`。正确做法是写一个 builder 函数，用循环生成：

```python
def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    attn_backend: str,
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits."""
    from torchtitan.models.common.config_utils import get_attention_config, make_ffn_config, make_gqa_config

    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            MyTransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    mask_type=mask_type,
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers
```

**为什么必须用 builder？**
- 每层 init 函数的 `std` 可能不同（depth-scaled）。
- 某些层可能是 MoE、某些层是 FFN（如 DeepSeek V3）。
- 不同尺寸的模型（8B/70B）可能只有 `dim/n_layers` 不同，结构完全一样，用 builder 可以避免复制粘贴。

#### 2.2.3 模型尺寸 builder：一个函数一个尺寸

```python
def _debugmodel(attn_backend: str) -> MyModel.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return MyModel.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
        ),
    )
```

**字段填写的硬性规则**：
- `tok_embeddings.num_embeddings` == `vocab_size`
- `lm_head.out_features` == `vocab_size`
- `lm_head.in_features` == `dim`
- `rope.dim` == `dim // n_heads`（如果使用标准 GQA）
- `rope.max_seq_len` 初始值可以设得很大，会在 `update_from_config` 中被 `seq_len` 覆盖

#### 2.2.4 `model_registry` 与 `ModelSpec`：每个字段的含义

```python
from torchtitan.protocols.model_spec import ModelSpec

__all__ = ["parallelize_mymodel", "MyModel", "mymodel_configs"]

def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = mymodel_configs[flavor](attn_backend=attn_backend)

    # 如果传入 converters（如 Float8, LoRA），按顺序应用
    if converters is not None:
        from torchtitan.models.utils import validate_converter_order
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)

    return ModelSpec(
        name="mymodel",              # 必须与文件夹名一致！
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_mymodel,  # 指向本目录 parallelize.py 中的函数
        pipelining_fn=pipeline_llm,          # 使用内置 PP，或 None
        post_optimizer_build_fn=None,        # MoE 模型可能需要 register_moe_load_balancing_hook
        state_dict_adapter=MyStateDictAdapter,  # 或 None
    )
```

`ModelSpec` 各字段详解：

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|----------|------|
| `name` | `str` | ✅ | 模型名，**必须和文件夹名一致**（因为 `--module` 参数会用来定位这个文件夹） |
| `flavor` | `str` | ✅ | 当前变体名（如 `"8B"`, `"debugmodel"`），仅用于日志和 checkpoint 命名 |
| `model` | `BaseModel.Config` | ✅ | 已经填好的模型配置树 |
| `parallelize_fn` | `Callable` | ✅ | 接收 `(model, *, parallel_dims, training, parallelism, compile_config, ac_config, dump_folder)`，返回并行化后的 model。**单卡也必须提供一个 no-op stub。** |
| `pipelining_fn` | `Callable \| None` | 可选 | PP 时使用 `pipeline_llm`，非 PP 时写 `None`。注意：如果 `parallelism.pipeline_parallel_degree > 1` 但这里为 `None`，Trainer 会直接报错。 |
| `post_optimizer_build_fn` | `Callable \| None` | 可选 | 优化器构建后执行的钩子。MoE 模型通常设置为 `register_moe_load_balancing_hook`，用于在优化器 step 前更新 `expert_bias`。 |
| `state_dict_adapter` | `type[BaseStateDictAdapter] \| None` | 可选 | 如果不加载/保存 HF checkpoint，写 `None`。 |

**为什么 `parallelize_fn` 不能为 `None`？**

Trainer 在非 PP 路径下会直接调用 `model_spec.parallelize_fn(model, ...)`。如果为 `None`，会报 `TypeError: 'NoneType' object is not callable`。即使你是单卡，也要写一个 `return model` 的 stub。

**`converters` 是什么？**

这是 torchtitan 支持的"Config 变换器"机制，用于在模型构建前修改 Config 树。常见 converter：
- `Float8LinearConverter`：把部分 Linear 替换为 Float8Linear
- `LoRAConverter`：在目标层注入 LoRA 低秩适配

`validate_converter_order` 会检查 converter 之间的依赖顺序（如 Float8 必须在 LoRA 之前）。

---

### 2.3 `config_registry.py` —— 定义训练配方（超参组合）

这个文件与模型结构**完全解耦**。它定义若干返回 `Trainer.Config` 的函数，运行时的 `--module mymodel --config mymodel_debugmodel` 就是通过 tyro CLI 选中这个函数。

#### 2.3.1 `Trainer.Config` 的关键字段

```python
from torchtitan.trainer import Trainer
from torchtitan.config import TrainingConfig, ParallelismConfig, ActivationCheckpointConfig, CompileConfig
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

def mymodel_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        # ===== 模型规范 =====
        # model_spec 被 tyro 抑制，不通过 CLI 传入，必须在 config_registry 中硬编码
        model_spec=model_registry("debugmodel"),

        # ===== Tokenizer =====
        # 可以指向任何包含 tokenizer 文件的目录
        hf_assets_path="./tests/assets/tokenizer",

        # ===== 损失函数 =====
        # ChunkedCELoss：把大的 vocab 拆成多段计算交叉熵，省显存
        # CrossEntropyLoss：标准实现（非 chunked），适合小 vocab 或调试
        loss=ChunkedCELoss.Config(),

        # ===== 优化器 =====
        optimizer=OptimizersContainer.Config(lr=8e-4),

        # ===== 学习率调度 =====
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),

        # ===== 训练超参 =====
        training=TrainingConfig(
            local_batch_size=8,   # 每张卡的 batch size
            seq_len=2048,         # 序列长度
            steps=10,             # 总训练步数
        ),

        # ===== 数据加载 =====
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",    # 或本地 json、parquet 等
        ),

        # ===== 并行配置 =====
        # 默认全为 1（单卡），需要并行时修改
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
        ),

        # ===== Checkpoint =====
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),

        # ===== 激活检查点 =====
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",     # "none", "selective", "full"
        ),

        # ===== Metrics / Logging =====
        metrics=MetricsProcessor.Config(log_freq=1),

        # ===== torch.compile =====
        compile=CompileConfig(enable=False),
    )
```

**几个容易忽视的字段**：

| 字段 | 默认值陷阱 |
|------|-----------|
| `parallelism` | 默认 `tensor_parallel_degree=1, pipeline_parallel_degree=1, ...`，但如果你要写一个 70B 的配置，记得在这里显式设置 `tensor_parallel_degree=8` 等 |
| `compile` | 默认关闭。如果要开，建议先确认模型能单卡跑通，再逐步开启 |
| `hf_assets_path` | 不要写成 HF Hub 的 model ID（如 `"meta-llama/Llama-3.1-8B"`），torchtitan 需要的是**本地路径**。下载方式见官方 README |

#### 2.3.2 变体之间的继承

不同训练配置之间可以通过 mutation 复用：

```python
def mymodel_debugmodel_flex_attn() -> Trainer.Config:
    config = mymodel_debugmodel()
    # 只改 model_spec，其余全部继承
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    return config

def mymodel_debugmodel_float8() -> Trainer.Config:
    from torchtitan.components.quantization import Float8LinearConverter
    config = mymodel_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[Float8LinearConverter.Config(model_compile_enabled=False)],
    )
    return config
```

---

### 2.4 `sharding.py` —— 声明式地描述"张量该怎么放"

**这是 torchtitan 最新架构中最关键也最容易被低估的文件。**

#### 2.4.1 核心任务

`sharding.py` 只做一件事：遍历模型的 Config 树，给每一个子模块的 Config 填上 `sharding_config: ShardingConfig`。

它**不**调用任何 `torch.distributed` API，它只是在 Config 上"贴标签"。真正的执行发生在 `parallelize.py` 中调用 `model.parallelize(parallel_dims)` 时。

#### 2.4.2 `ShardingConfig` 的六个字段逐字段详解

```python
from torchtitan.protocols.sharding import ShardingConfig, LocalMapConfig
from torchtitan.protocols.types import MeshAxisName

ShardingConfig(
    state_shardings={...},    # 参数/Buffer 怎么分
    in_src_shardings={...},   # forward 输入当前在哪
    in_dst_shardings={...},   # forward 输入应该去哪
    out_src_shardings=...,    # forward 输出当前在哪
    out_dst_shardings=...,    # forward 输出应该去哪
    local_map=...,            # 是否用 local_map 包装
)
```

**`state_shardings`: `dict[str, NamedPlacement]`**

Key 是参数名（如 `"weight"`, `"bias"`），Value 描述该参数在 mesh 各轴上的 Placement。

```python
from torch.distributed.tensor import Replicate, Shard
from torchtitan.models.common.decoder_sharding import dense_param_placement

# 示例：ColwiseParallel 的 weight 是 Shard(0)
state_shardings={
    "weight": dense_param_placement(tp=Shard(0)),
    "bias": dense_param_placement(tp=Shard(0)),
}
```

`dense_param_placement(tp=...)` 是一个 helper，它返回：

```python
{
    MeshAxisName.DP_REPLICATE: Replicate(),
    MeshAxisName.DP_SHARD: Replicate(),
    MeshAxisName.CP: Replicate(),
    MeshAxisName.TP: Shard(0),   # 你传入的 tp
}
```

含义：在 DP 轴上复制参数，在 TP 轴上按第 0 维切分。FSDP 会在 `parallelize` 之后进一步把 `DP_SHARD` 轴上的复制参数切成 shard。

**`in_src_shardings` 与 `in_dst_shardings`: `dict[str, NamedPlacement]`**

Key 是 `forward` 的参数名（通过 `inspect.signature(forward)` 获取，**注意必须是参数名，不能错**）。

典型场景（Sequence Parallel）：

```python
ShardingConfig(
    in_src_shardings={"x": dense_activation_placement(tp=Shard(1))},
    in_dst_shardings={"x": dense_activation_placement(tp=Replicate())},
)
```

含义：输入 `x` 当前在 TP 轴上是 `Shard(1)`（seq 维度切分），进入 `forward` 前需要 all-gather 成 `Replicate()`。

`dense_activation_placement` 与 `dense_param_placement` 的区别：
- `dense_activation_placement` 在 DP 轴上是 `Shard(0)`（batch 维度切分），因为数据并行时每个 rank 处理不同 batch。
- `dense_param_placement` 在 DP 轴上是 `Replicate()`，因为参数在 FSDP 之前是复制的。

**`out_src_shardings`: `NamedPlacement | tuple[NamedPlacement, ...] | None`**

描述 `forward` 输出本身的 placement。主要用于 `local_map` 回包。

例如 Attention 的 `inner_attention` 输出 `Partial()`（Rowwise matmul 的结果），需要被 MoE wrapper 或上一层 reduce：

```python
out_src_shardings=dense_activation_placement(tp=Partial())
```

**`out_dst_shardings`: `NamedPlacement | None`**

描述 `forward` 结束后，输出应该被重分布到哪里。

例如 Attention 的 `wo` 是 Rowwise，输出需要 reduce-scatter 到 seq-parallel 布局：

```python
out_dst_shardings=dense_activation_placement(tp=Shard(1) if enable_sp else Replicate())
```

**`local_map`: `LocalMapConfig | None`**

当模块内部使用**不支持 DTensor 的自定义 kernel** 时（如 FlashAttention、DeepEP、 grouped_mm），必须用 `local_map` 把 DTensor 解包成 local tensor，kernel 计算完后再包回 DTensor。

```python
LocalMapConfig(
    in_grad_placements=(
        dense_activation_placement(tp=Shard(2)),      # q 的 grad
        dense_activation_placement(tp=Shard(2), cp=Partial()),  # k 的 grad
        dense_activation_placement(tp=Shard(2), cp=Partial()),  # v 的 grad
    )
)
```

`in_grad_placements` 描述了 backward 时输入梯度的期望 placement。`local_map` 会据此在反向时做正确的 redistribute。

#### 2.4.3 `NamedPlacement` 与 mesh axis

`NamedPlacement = dict[MeshAxisName, Placement]`，本质上是给每个 mesh 维度命名后的 placement 描述。

torchtitan 中涉及的所有 mesh axis：

| Axis Name | 全称 | 作用域 |
|-----------|------|--------|
| `dp_replicate` | Data Parallel Replicate | DDP/HSDP 的复制维度 |
| `dp_shard` | Data Parallel Shard | full_dtensor 下的显式 DP shard 轴 |
| `fsdp` | FSDP | legacy 路径下，`dp_shard * cp` 被折叠成的轴 |
| `cp` | Context Parallel | CP 的序列切分轴 |
| `tp` | Tensor Parallel | TP 的 hidden/head 切分轴 |
| `ep` | Expert Parallel | MoE 的 expert 切分轴 |
| `efsdp` | Expert FSDP | MoE 参数上的 FSDP 轴，等于 `fsdp * tp // ep` |

**dense 路径**（attention、FFN、norm、embed）使用 dense mesh：
- `full_dtensor=True` 时：`(dp_replicate, dp_shard, cp, tp)`
- legacy 时：`(dp_replicate, fsdp, tp)`

**sparse 路径**（MoE expert 参数）使用 sparse mesh：
- `(dp_replicate, efsdp, ep)`

**Placement 类型**：
- `Replicate()`: 每个 rank 持有完整副本
- `Shard(dim)`: 在 `dim` 维度上切分
- `Partial()`: 每个 rank 持有部分和，需要 reduce

#### 2.4.4 `LocalMapConfig` 什么时候必须用

| 场景 | 是否需要 `local_map` | 原因 |
|------|---------------------|------|
| `nn.Linear` / `nn.RMSNorm` | ❌ | PyTorch 原生支持 DTensor |
| SDPA / FlexAttention / FlashAttention | ✅ | Attention kernel 内部不做 DTensor 通信，需要 local tensor |
| `torch._grouped_mm` (MoE experts) | ✅ | grouped_mm 不支持 DTensor |
| DeepEP / AllToAll token dispatch | ✅ | 自定义 CUDA kernel |
| 普通 elementwise op (silu, mul, add) | ❌ | DTensor 自动传播 |

#### 2.4.5 复用 `models/common/decoder_sharding.py` 的 helper

不要从零手写每个 ShardingConfig。`decoder_sharding.py` 提供了一系列标准 helper：

```python
from torchtitan.models.common.decoder_sharding import (
    colwise_config,           # ColwiseParallel: weight Shard(0), output Shard(-1)
    rowwise_config,           # RowwiseParallel: weight Shard(1), output Replicate/Shard(1)
    norm_config,              # Norm: weight Replicate, input/output 根据 enable_sp
    set_decoder_sharding_config,      # 设置 tok_embeddings, norm, lm_head
    set_dense_ffn_sharding,           # 设置 FeedForward 的 w1/w2/w3
    set_gqa_attention_sharding,       # 设置 Attention 的 qkv/wo
    set_gqa_inner_attention_local_map, # 设置 inner_attention 的 local_map
)
```

一个典型的 `sharding.py` 只需几十行：

```python
from torchtitan.models.common.decoder_sharding import (
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)

def set_mymodel_sharding_config(config, *, loss_parallel: bool, enable_sp: bool) -> None:
    # 1. 设置根层（tok_embeddings, norm, lm_head, freqs_cis）
    set_decoder_sharding_config(config, loss_parallel=loss_parallel, enable_sp=enable_sp)

    # 2. 遍历每层，设置 attention 和 ffn
    for layer_cfg in config.layers:
        norm = norm_config(enable_sp=enable_sp)
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm

        set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
        set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

        assert layer_cfg.feed_forward is not None
        attn_x_placement = Shard(1) if enable_sp else Replicate()
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )
```

#### 2.4.6 调用时机

`set_<model>_sharding_config` 通常在 `Model.Config.update_from_config` 中被调用：

```python
def update_from_config(self, *, trainer_config, **kwargs):
    # ... 其他检查和同步 ...
    from .sharding import set_mymodel_sharding_config
    set_mymodel_sharding_config(
        self,
        loss_parallel=not trainer_config.parallelism.disable_loss_parallel,
        enable_sp=trainer_config.parallelism.enable_sequence_parallel,
    )
```

这样 sharding 配置就能根据训练时的并行设置动态调整（例如用户开了 SP，就设置 `enable_sp=True`）。

---

### 2.5 `parallelize.py` —— 编排并行化与训练技巧的执行顺序

这个文件是"指挥官"。它接收一个已经构建好的 `nn.Module` 实例，按**固定顺序**施加各种变换。

#### 2.5.1 标准执行顺序与原因

```
parallelize_<model>(model, parallel_dims, training, parallelism, ...)
  ├─ 1. CP 包装（如果需要）          apply_cp_to_forward()
  ├─ 2. TP/SP/EP 切分               model.parallelize(parallel_dims)
  ├─ 3. 可选：异步 TP               maybe_enable_async_tp()
  ├─ 4. 激活检查点                  apply_ac()
  ├─ 5. torch.compile               apply_compile()
  └─ 6. FSDP / HSDP                 apply_fsdp()
```

**为什么不能乱序？**

1. **CP 必须在 `parallelize()` 之前**：`apply_cp_to_forward` 包装的是 `inner_attention.forward`，而 `parallelize()` 会把 `forward` 替换为 `forward_with_redistribution`。如果先 `parallelize()`，`apply_cp_to_forward` 就找不到原始的 `inner_attention.forward` 了。

2. **TP 必须在 AC 之前**：`apply_ac` 会包一层 `CheckpointWrapper`，改变 module 层级。如果先 AC 再 TP，`parallelize()` 递归时可能找不到真正的 `Module` 子节点。

3. **Compile 通常在 FSDP 之前**：某些 backend 要求 `torch.compile` 看到原始的 module 结构，而不是 FSDP 包裹后的结构。

4. **FSDP 必须在最后**：`fully_shard` 安装 `__call__` hook，会在每次 forward/backward 前后做 all-gather/reduce-scatter。如果前面有 AC 或 compile，FSDP 的 hook 需要包在最外层。

#### 2.5.2 `apply_fsdp` 详解

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torchtitan.distributed.fsdp import disable_fsdp_gradient_division, get_fsdp_reshard_after_forward_policy

def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    dp_mesh_dims=None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,    # 如 torch.bfloat16
        reduce_dtype=reduce_dtype,  # 如 torch.bfloat16
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    # 对 tok_embeddings 单独 FSDP
    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard_after_forward)

    # norm + lm_head 可以放在同一个 FSDP 单元中
    if model.norm is not None and model.lm_head is not None:
        fully_shard([model.norm, model.lm_head], **fsdp_config, reshard_after_forward=reshard_after_forward)

    # 每层 TransformerBlock 单独 FSDP
    for layer_id, block in model.layers.items():
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)

    # 最后对整个 model FSDP（形成层级结构）
    fully_shard(model, **fsdp_config)

    # 关闭 FSDP 自动梯度除法（torchtitan 自己控制）
    disable_fsdp_gradient_division(model)
```

**关键点**：

- `fully_shard` 的调用顺序是从**叶子到根**：先叶子模块（tok_embeddings, block, norm/lm_head），最后根模块 `model`。这形成一个层级 FSDP 结构。
- 如果 `enable_weight_tying`，`tok_embeddings` 和 `lm_head` 共享同一个参数。此时应该把它们放在**同一个 `fully_shard` 调用**中（传 list），避免重复 all-gather。
- `reshard_after_forward` 控制 forward 后是否释放参数 shard。默认策略对最后一个 stage 的 norm+lm_head 会做优化（不 reshard，因为马上又要 prefetch）。

#### 2.5.3 完整代码模板

```python
from torchtitan.distributed.fsdp import apply_fsdp  # 或从 llama3/llama4 复制

def parallelize_mymodel(
    model: MyModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    # 1. 检查 seq_len 可除性
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"seq_len {training.seq_len} must be divisible by tp*2*cp"
    )

    # 2. full_dtensor 额外校验
    if parallelism.full_dtensor:
        from torchtitan.distributed.full_dtensor import validate_config, resolve_fsdp_mesh
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        # 非 full_dtensor：CP 在 parallelize 之前
        if parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import apply_cp_to_forward
            apply_cp_to_forward(
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        # TP 才需要 model.parallelize()
        if parallel_dims.tp_enabled:
            model.parallelize(parallel_dims)

    # 3. 异步 TP（可选优化）
    if parallel_dims.tp_enabled:
        from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # 4. 激活检查点
    if ac_config.mode != "none":
        from torchtitan.distributed.activation_checkpoint import apply_ac
        apply_ac(model, ac_config, model_compile_enabled=False, base_folder=dump_folder)

    # 5. torch.compile
    if compile_config.enable and "model" in compile_config.components:
        from torchtitan.distributed.compile import apply_compile
        apply_compile(model, compile_config)

    # 6. FSDP
    if parallelism.full_dtensor:
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None

    apply_fsdp(
        model, dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
    )

    return model
```

---

### 2.6 `state_dict_adapter.py` —— HF 格式互转（可选）

#### 2.6.1 什么时候需要

| 场景 | 是否需要 |
|------|----------|
| 从头训练，不需要加载任何 pretrained checkpoint | ❌ |
| 加载 HuggingFace checkpoint 做继续预训练 / 微调 | ✅ |
| 训练完后要保存为 HF 格式供 vLLM/Transformers 推理 | ✅ |
| 做数值对齐测试（torchtitan vs HF） | ✅ |

#### 2.6.2 实现模板

```python
import re
from typing import Any
import torch
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .model import MyModel

class MyModelStateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: MyModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        # 定义 HF -> native 的映射表
        # None 表示该 key 在目标格式中不存在（如 HF 的 rope inv_freq）
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """native -> HF"""
        # 反转映射表
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}
        for key, value in state_dict.items():
            if "layers" in key:
                # 把具体层号替换为 {} 以便查表
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map.get(key)
            if new_key is not None:
                hf_state_dict[new_key] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """HF -> native"""
        state_dict = {}
        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
            if new_key is not None:
                state_dict[new_key] = value
        return state_dict
```

**`StateDictAdapter` vs `BaseStateDictAdapter`**：

- `BaseStateDictAdapter` 是纯抽象基类，只定义接口。
- `StateDictAdapter` 提供了默认实现，包括：
  - 自动读取 `model.safetensors.index.json` 构建 `fqn_to_index_mapping`
  - 提供 `fused_to_separate_qkv` / `separate_to_fused_qkv` 的静态方法
  - 提供默认的 `get_hf_storage_reader`

**绝大多数模型都应该继承 `StateDictAdapter`**，而不是 `BaseStateDictAdapter`。

**注意**：如果用了 `fuse_qkv`，HF 格式是分开的 `q_proj, k_proj, v_proj`，而 native 格式是融合的 `wqkv`。这时需要在 `from_hf` 中收集 Q/K/V 然后 fuse，在 `to_hf` 中 split。`StateDictAdapter` 基类已经提供了 `fused_to_separate_qkv` 和 `separate_to_fused_qkv` 静态方法可以直接用。

---

## 3. 最小可行模型与并行扩展路径

> 本节回答："我想要一个最简单、能单卡跑起来的新模型，最少要准备几个文件？之后每增加一种并行策略，又要新增/修改什么？"

### 3.1 单卡 MVP（4 个文件 + 1 处修改）

**文件清单**：

| 文件 | 是否必须 | 说明 |
|------|----------|------|
| `model.py` | ✅ | 定义 `MyModel(Decoder)` 和 `MyTransformerBlock(TransformerBlock)` |
| `__init__.py` | ✅ | 组装 Config，注册 `ModelSpec`，`parallelize_fn` 写成一个 no-op |
| `config_registry.py` | ✅ | 定义训练配置，指定 tokenizer 路径（可指向其他模型的 tokenizer） |
| `models/__init__.py` | ✅ | 把模型名加入 `_supported_models` |
| `parallelize.py` | 推荐 | 如果只有单卡，可以省；但通常放一个 no-op 或只留 `return model` 的 stub |
| `sharding.py` | ❌ | 单卡完全不需要 |
| `state_dict_adapter.py` | ❌ | 不需要 HF 互转时可省 |

**最小的 `parallelize_fn` stub**：

```python
def parallelize_mymodel(model, *, parallel_dims, training, parallelism,
                        compile_config, ac_config, dump_folder):
    # 单卡：什么都不做，直接返回
    return model
```

**为什么可以这么少？** 因为 torchtitan Trainer 的核心循环（数据加载、前反向、优化器 step、checkpoint 保存）与模型结构是解耦的。只要你的模型满足 `Decoder` 的接口契约，Trainer 就能驱动它训练。

### 3.2 扩展到 DDP / FSDP（+1 个文件，修改 2 个文件）

DDP/FSDP 的本质是**数据并行**：把同一份模型参数复制到多张卡，每张卡处理不同 micro-batch，然后 all-reduce / reduce-scatter 梯度。

**新增/修改清单**：

| 变更项 | 内容 |
|--------|------|
| `parallelize.py` | 引入 `apply_fsdp()`（可直接复制 llama3 的 `apply_fsdp` 辅助函数） |
| `model.py` 的 `Config` | 在 `update_from_config()` 中**可选地**调用 `set_sharding_config` |

**重要**：纯 FSDP（不开启 TP/SP/EP/CP）**仍然不需要 `sharding.py`**！

原因：FSDP 是在 `fully_shard` 的 `__call__` 层级做参数分片，不需要 `ShardingConfig` 中那种对 `forward` 输入输出激活的精细控制。`apply_fsdp` 直接对 `model.tok_embeddings`、每个 `TransformerBlock`、`model.norm`、`model.lm_head` 调用 `fully_shard(...)` 即可。

**但有个陷阱**：如果你后续还要叠加 TP，那么 FSDP 和 TP 的交互就需要 `sharding_config` 来协调了。所以官方实现（llama3/qwen3 等）总是**预先**把 `sharding.py` 写好，即使当前只跑 FSDP。

### 3.3 扩展到 Tensor Parallel / Sequence Parallel（+1 个文件，修改 2 个文件）

TP/SP 进入**张量并行**领域：同一层的参数被切分到不同卡上，每张卡只存一部分权重，前向时通过 all-gather / reduce-scatter 通信。

**必须引入 `sharding.py`**。

原因：TP/SP 需要精确控制：
1. 每个 `Linear.weight` 在 TP mesh 上是 `Shard(0)`（Colwise）还是 `Shard(1)`（Rowwise）
2. 每层 `forward` 的输入/输出激活是否需要从 `Replicate` 重分布到 `Shard(1)`（SP）
3. Attention 内部的 `q, k, v` 在进 kernel 前需要 `local_map` 解开 DTensor

**修改清单**：

| 变更项 | 内容 |
|--------|------|
| 新增 `sharding.py` | 为 embedding、norm、attention、ffn、lm_head 设置 `ShardingConfig` |
| 修改 `model.py` 的 `Config.update_from_config` | 调用 `set_<model>_sharding_config(self, loss_parallel=..., enable_sp=...)` |
| 修改 `parallelize.py` | 在合适位置加入 `model.parallelize(parallel_dims)` |

**TP 对模型维度的硬性要求**：
- `n_heads % tp_degree == 0`
- `n_kv_heads % tp_degree == 0`（如果 `n_kv_heads` 不为 None）
- `seq_len % tp_degree == 0`（如果开启了 Sequence Parallel）

这些检查通常写在 `update_from_config()` 中，在模型构建**之前**抛出清晰的错误。

### 3.4 扩展到 Pipeline Parallel（修改 2 个文件）

PP 把**不同层**放到不同卡上（或不同节点上）。

**修改清单**：

| 变更项 | 内容 |
|--------|------|
| `__init__.py` 的 `ModelSpec` | `pipelining_fn=pipeline_llm`（使用框架内置函数即可） |
| `model.py` 的 `Decoder.forward` | 确保 `tok_embeddings`、`norm`、`lm_head` 为 `None` 时做 passthrough（`pipeline_llm` 会 prune 掉非本 stage 的模块） |

torchtitan 的 `pipeline_llm` 会自动按层数切分 stage，不需要你手动写 PP 切分逻辑。但要支持 weight tying 的模型需要注意：`enable_weight_tying=True` 与 PP 不兼容（因为 embedding 和 lm_head 可能在不同 stage）。这个检查也通常放在 `update_from_config()` 中。

### 3.5 扩展到 Expert Parallel（+修改 `model.py` 和 `sharding.py`）

EP 只针对 **MoE 层**。如果你的模型有 `MoE`（而不是 `FeedForward`），则：

1. **在 `model.py` 的 `TransformerBlock` 中支持 `moe` 字段**
   - `TransformerBlock.Config` 本身就有 `feed_forward` 和 `moe` 两个可选字段，你可以根据层号决定某些层用 `feed_forward`、某些层用 `moe`。

2. **`sharding.py` 中引入稀疏 mesh 的 placement**
   - 使用 `models/common/moe_sharding.py` 中的 `set_moe_sharding_config(...)`
   - Routed expert 的参数使用 sparse mesh（轴为 `dp_replicate, efsdp, ep`），与 dense 部分（`dp_replicate, dp_shard, cp, tp`）不同

3. **`parallelize.py` 可能需要调用 `apply_moe_ep_tp`**
   - 目前 MoE 的 EP+TP 混合并行还有部分逻辑不在 `ShardingConfig` 中，需要在 `parallelize.py` 里额外调用模型特定函数

### 3.6 扩展到 Context Parallel（通常只需修改 `parallelize.py`）

CP 把**序列维度**切分到不同卡上，主要在 Attention 层做 ring attention。

torchtitan 的 CP 支持是通过 `apply_cp_to_forward()` 包装 Attention 的 `inner_attention.forward` 实现的，**不需要修改模型结构或 sharding config**。只需确保：

- `seq_len % (2 * cp_degree) == 0`
- 不搭配 `VarlenAttention`（目前 CP 只支持 SDPA 和 FlexAttention）

### 3.7 扩展路径总结表

| 目标 | 最少新增文件 | 关键修改点 |
|------|-------------|-----------|
| **单卡 MVP** | 3 个（`model.py`, `__init__.py`, `config_registry.py`） | `models/__init__.py` 加模型名；`parallelize_fn` 可 stub |
| **+ DDP/FSDP** | 0（复用 `parallelize.py` stub 扩展） | `parallelize.py` 中加入 `apply_fsdp()` |
| **+ TP/SP** | 1 个（`sharding.py`） | `update_from_config()` 调用 sharding；`parallelize.py` 调用 `model.parallelize()` |
| **+ PP** | 0 | `ModelSpec.pipelining_fn=pipeline_llm`；`forward` 支持 passthrough |
| **+ CP** | 0 | `parallelize.py` 中 `apply_cp_to_forward()`；检查 `seq_len` 可除性 |
| **+ EP (MoE)** | 0（修改 `sharding.py`） | 模型层改用 `MoE.Config`；sharding 引入 `set_moe_sharding_config` |

---

## 4. parallel.py 与 sharding.py 的配合关系

这是 torchtitan 并行架构中最精妙的设计之一，也是新手最容易困惑的地方。我们用一张图和一段代码来说明。

### 4.1 职责划分

```
sharding.py
    │   "声明" —— 在 Config 阶段运行
    │   给每个子模块贴标签：weight 该 Shard(0) 还是 Shard(1)，
    │   forward 输入该 Replicate 还是 Shard(1) ...
    ▼
Config 树（每个 Module.Config 都有 sharding_config）
    │
    │   build()
    ▼
nn.Module 实例（尚未做任何并行化）
    │
    │   "执行" —— 在 Runtime 阶段运行
    ▼
parallelize.py
    ├─ model.parallelize(parallel_dims)  ← 通用引擎，递归所有 Module
    │     ├─ _shard_states()            ← 把 param/buffer distribute_tensor
    │     ├─ _redistribute_inputs()     ← 对 forward 输入做 from_local / redistribute
    │     ├─ [可选] local_map           ← 把 DTensor 解包成 local tensor
    │     ├─ 原始 forward()
    │     ├─ _redistribute_outputs()    ← 对 forward 输出做 redistribute
    │     └─ 返回 DTensor
    ├─ apply_fsdp()
    └─ ...
```

**一句话总结**：
- `sharding.py` 决定**怎么切**（策略声明）。
- `parallelize.py` 决定**什么时候切、按什么顺序切**（编排执行）。
- `Module.parallelize()`（在 `protocols/module.py`）才是**真正动刀**的人（通用执行引擎）。

### 4.2 一个具体例子：Llama3 Attention 的 TP 切分

**在 `sharding.py` 中**，我们为 Attention 设置：

```python
# set_gqa_attention_sharding() 的逻辑简化
attention_cfg.sharding_config = ShardingConfig(
    in_src_shardings={
        "x": dense_activation_placement(tp=Shard(1) if enable_sp else Replicate()),
        "rope_cache": dense_param_placement(tp=Replicate()),
    },
    in_dst_shardings={
        "x": dense_activation_placement(tp=Replicate()),  # all-gather
        "rope_cache": dense_param_placement(tp=Replicate()),
    },
)
# qkv_linear 的 weight 是 Colwise (Shard(0))
attention_cfg.qkv_linear.sharding_config = colwise_config()
# wo 的 weight 是 Rowwise (Shard(1))，输出根据 enable_sp 决定 all-reduce 或 reduce-scatter
attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)
```

**在 `parallelize.py` 中**，我们只需一行：

```python
if parallel_dims.tp_enabled:
    model.parallelize(parallel_dims)
```

**在 `Module.parallelize()` 中**，引擎会：

1. 先递归调用所有子模块的 `parallelize()`（后序遍历，确保叶子节点先被切分）。
2. 对当前模块的参数，根据 `state_shardings` 调用 `distribute_tensor(param, mesh, placements)`。
   - 例如 `qkv_linear.wqkv.weight` 变成 `DTensor(mesh=tp_mesh, placements=(Shard(0),))`。
3. 把 `forward` 替换成 `forward_with_redistribution`：
   - 输入 `x` 如果是 plain tensor，先用 `DTensor.from_local` 包成 DTensor（按 `in_src_shardings`）。
   - 然后按 `in_dst_shardings` 做 `redistribute`（如从 `Shard(1)` 变成 `Replicate()`，即 all-gather）。
   - 调用原始 `forward`。
   - 输出按 `out_dst_shardings` 做 `redistribute`（如从 `Partial()` 变成 `Shard(1)`，即 reduce-scatter）。

**为什么这样设计？** 因为不同的并行策略（TP、SP、EP）对同一个层的切分逻辑是**正交但可组合**的。`ShardingConfig` 用 `NamedPlacement`（`dict[MeshAxisName, Placement]`）描述在所有 mesh 轴上的布局，而 `resolve_placements()` 在运行时根据实际启用的 mesh 轴过滤出有效的 placement。这意味着：

- 单卡运行时，mesh 所有轴大小都是 1，`resolve_mesh` 返回 None，`parallelize()` 几乎什么都不做（`Shard(d)` 在 size-1 mesh 上会被归一化为 `Replicate()`）。
- 只开 TP 时，`DP_REPLICATE/DP_SHARD/CP` 的 placement 被忽略，只消费 `TP` 轴。
- 开 `full_dtensor` 时，所有轴同时生效，DTensor 是一个真正的多维 sharded tensor。

### 4.3 什么时候 `parallelize.py` 需要额外干预？

虽然大部分 TP/SP 逻辑可以由 `ShardingConfig` 自动处理，但以下几种情况 `parallelize.py` 需要手动补充：

1. **CP 包装**：CP 是在 Attention 的 `inner_attention.forward` 外再包一层 ring attention 逻辑，目前不是通过 `ShardingConfig` 实现的，而是直接调用 `apply_cp_to_forward()`。
2. **MoE EP+TP**：目前部分 MoE 专家并行的代码还未完全迁移到 `ShardingConfig`，需要在 `parallelize.py` 中额外调用 `apply_moe_ep_tp()`。
3. **FSDP/HSDP**：这是参数存储策略，不修改 forward 的输入输出，所以不走 `ShardingConfig`，而是直接调用 `fully_shard()`。
4. **AC / Compile**：与并行无关，纯粹是训练优化，必须在 TP 之后、FSDP 之前施加。

---

## 5. Layer Config 组装时的维度契约与检查机制

> 本节回答："在 `__init__.py` 里把 Embedding、Attention、FFN、Norm、LM Head 的 Config 拼成一棵配置树时，层与层之间的维度需要满足什么条件？不满足时 torchtitan 会怎么报错？"

### 5.1 标准数据流与维度契约

以一个标准的 Decoder-only Transformer 为例，数据流和维度如下：

```
tokens:            [batch_size, seq_len]           (int64)
  │
  ▼
tok_embeddings:    [batch_size, seq_len, dim]      (float)
  │
  ▼
TransformerBlock × N
  ├── attention_norm:  [batch_size, seq_len, dim] → [batch_size, seq_len, dim]
  ├── attention
  │     ├── qkv_linear:  [B, S, dim] → [B, S, (n_heads + 2*n_kv) * head_dim]
  │     │                    然后 reshape 为 [B, S, n_heads, head_dim] (q)
  │     │                               和 [B, S, n_kv_heads, head_dim] (k, v)
  │     ├── inner_attention: [B, S, n_heads, head_dim] × 3 → [B, S, n_heads, head_dim]
  │     └── wo:  [B, S, n_heads * head_dim] → [B, S, dim]
  ├── 残差加: [B, S, dim] + [B, S, dim]
  ├── ffn_norm:  [B, S, dim] → [B, S, dim]
  └── feed_forward
        ├── w1:  [B, S, dim] → [B, S, hidden_dim]
        ├── w3:  [B, S, dim] → [B, S, hidden_dim]
        ├── silu + multiply
        └── w2:  [B, S, hidden_dim] → [B, S, dim]
  └── 残差加: [B, S, dim] + [B, S, dim]
  │
  ▼
norm:              [batch_size, seq_len, dim] → [batch_size, seq_len, dim]
  │
  ▼
lm_head:           [batch_size, seq_len, dim] → [batch_size, seq_len, vocab_size]
```

**层间隐式维度契约**：

| 连接点 | 契约条件 | 违反后果 |
|--------|----------|----------|
| Embedding.out → Attention.in | `embedding_dim == dim` | PyTorch RuntimeError: matmul shape mismatch |
| Attention.qkv → inner_attention | qkv reshape 后的 `head_dim * n_heads` 必须等于 `dim` | `view()` 时元素总数不匹配 |
| Attention.inner → wo | `wo.in_features == n_heads * head_dim` | Linear 输入特征数不匹配 |
| Attention.wo.out → FFN.in | `wo.out_features == dim` | matmul shape mismatch |
| FFN.w2.out → next_block.in | `w2.out_features == dim` | matmul shape mismatch |
| Norm.out → lm_head.in | `lm_head.in_features == dim` | Linear 输入特征数不匹配 |

### 5.2 torchtitan 中已有的显式检查

torchtitan 并没有在所有连接点上都做前置静态检查（因为 Config 树的灵活性很高，很多维度是推导出来的），但它在**关键决策点**做了检查：

**A. Attention Config 自检查（`BaseAttention.Config.__post_init__`）**

```python
assert self.n_heads > 0, "n_heads must be > 0"
assert self.mask_type in ["causal", "block_causal"], ...
if isinstance(self.inner_attention, ScaledDotProductAttention.Config) and self.mask_type == "block_causal":
    raise ValueError("...")
```

**B. Llama3Model Config 的并行度检查（`update_from_config`）**

```python
tp = parallelism.tensor_parallel_degree
if tp > 1:
    n_heads = self.layers[0].attention.n_heads
    n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
    if n_heads % tp != 0:
        raise ValueError(f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads}).")
    if n_kv_heads % tp != 0:
        raise ValueError(f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads}).")

if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
    raise NotImplementedError("Weight tying is not supported with Pipeline Parallel.")
```

**C. `parallelize.py` 中的序列长度检查**

```python
assert training.seq_len % parallel_dims.seq_len_divisor == 0, ...
# 其中 seq_len_divisor = tp * (cp * 2)
```

**D. `Module._shard_states()` 中的参数完整性检查**

```python
for name, param in self.named_parameters(recurse=False):
    named_placements = sharding_config.state_shardings.get(name)
    if named_placements is None:
        raise ValueError(f"{type(self).__name__}.{name} has no placement declared ...")
```

这意味着：如果你开了 TP，但忘了给某个 `Linear` 的 `bias` 写 `sharding_config`，`parallelize()` 会立刻报错，而不是在训练中途静默失败。

**E. `resolve_placements()` 中的 mesh axis 完备性检查**

```python
for i, axis_name in enumerate(mesh.mesh_dim_names):
    key = MeshAxisName(axis_name)
    if key not in named:
        raise ValueError(
            f"ShardingConfig does not declare a placement for mesh axis {axis_name!r}. "
            f"Declared: {sorted(k.value for k in named)}; "
            f"required: {list(mesh.mesh_dim_names)}."
        )
```

这在 `full_dtensor=True` 时尤其重要：所有 `ShardingConfig` 必须为 dense mesh 的 `dp_replicate, dp_shard, cp, tp` 或 sparse mesh 的 `dp_replicate, efsdp, ep` 中的**每一个轴**声明 placement。漏掉任何一个轴都会在这里被拦截。

**F. 运行时 PyTorch shape 检查（最后一道防线）**

如果以上所有静态检查都通过了，但在 layer config 组装时你写错了维度（例如 `wo.in_features` 比 `n_heads * head_dim` 小 1），那么：

- 在 `model_config.build()` 时不会报错（Linear 的 `in_features/out_features` 只是记录数字，不验证）。
- 在 `init_states()` 时也不会报错（参数初始化不验证形状兼容性）。
- **第一次 `forward()` 时**，PyTorch 的 `F.linear` 或 `matmul` 会抛出 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (...)`。

这就是 torchtitan 的设计哲学：**Config 层面的检查只覆盖"结构性/并行性"约束；数值维度的匹配主要依赖 PyTorch 的运行时检查。** 好处是代码简洁，不重复造轮子；代价是你在第一次 forward 之前不会知道维度拼错了。

### 5.3 如何避免维度错误？最佳实践

1. **复用 `config_utils.py` 中的 builder**：
   - `make_gqa_config(dim=..., n_heads=..., ...)` 会自动计算 `wo.in_features = n_heads * per_head_dim` 和 `wo.out_features = dim`，避免手滑。
   - `make_ffn_config(dim=..., hidden_dim=...)` 会自动设置 `w1.in=dim, w1.out=hidden_dim, w2.in=hidden_dim, w2.out=dim`。

2. **在自定义 builder 中写 assert**：
   ```python
   def _build_layers(...):
       assert hidden_dim % tp == 0, "hidden_dim must be divisible by tp"
       ...
   ```

3. **用 debug model 做冒烟测试**：
   先写一个极小模型（`dim=256, n_layers=2, vocab_size=1024`），单卡跑一个 step，确认 forward/backward 都能过，再放大到目标尺寸。

---

## 6. Tokenizer 复用与无 HF 仓模型

如果你不想为新模型单独准备 HuggingFace tokenizer，可以直接在 `config_registry.py` 中借用其他模型的：

```python
def mymodel_debug() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",  # 指向 llama3 的测试 tokenizer
        model_spec=model_registry("debugmodel"),
        ...
    )
```

`hf_assets_path` 只需要是一个包含 `tokenizer.json`（或对应 tokenizer 所需文件）的目录。Trainer 在初始化 dataloader 时会从中加载 tokenizer。因此：

- **不需要** `tokenizer.py`。
- **不需要** 在 HuggingFace 上创建仓库。
- 只要 tokenizer 的 `vocab_size` 与你的 `model.py` 中 `Config.vocab_size` 一致即可。

---

## 7. 完整示例：从 Llama3 出发做减法

假设你想定义一个叫 `tiny_moe` 的模型，结构是：标准 Llama3 Decoder，但把第 2、4 层换成 MoE，其余层保持 FFN。最小文件集合如下：

### 7.1 `model.py`

```python
from dataclasses import dataclass
import torch
from torchtitan.models.common import TransformerBlock
from torchtitan.models.common.decoder import Decoder

class TinyMoETransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass  # 空继承即可，父类已有 attention/feed_forward/moe 字段

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        # 根据 config 决定是 feed_forward 还是 moe
        if config.moe is not None:
            self.feed_forward = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(self, x, freqs_cis, attention_masks, positions=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, attention_masks, positions)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TinyMoEModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 512
        vocab_size: int = 8192

        def update_from_config(self, *, trainer_config, **kwargs):
            # 可在此调用 set_tinymoe_sharding_config(...) 如果要做 TP
            pass

        def get_nparams_and_flops(self, model, seq_len):
            # 粗略估算即可
            from torchtitan.models.utils import get_dense_model_nparams_and_flops
            return get_dense_model_nparams_and_flops(
                model, n_layers=len(self.layers), n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len, enable_weight_tying=False,
            )

    def __init__(self, config: Config):
        super().__init__(config)
```

### 7.2 `__init__.py`

```python
from functools import partial
import torch.nn as nn
from torchtitan.models.common import Embedding, RMSNorm, Linear, TransformerBlock
from torchtitan.models.common.config_utils import get_attention_config, make_ffn_config, make_gqa_config
from torchtitan.models.common.feed_forward import compute_ffn_hidden_dim
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.protocols.model_spec import ModelSpec
from .model import TinyMoEModel, TinyMoETransformerBlock

def _build_layers(n_layers, dim, n_heads, hidden_dim, attn_backend):
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        # 第 2、4 层放 MoE，其余放 FFN（示例）
        use_moe = (layer_id + 1) in {2, 4}
        ffn_cfg = None if use_moe else make_ffn_config(
            dim=dim, hidden_dim=hidden_dim,
            w1_param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
            w2w3_param_init={"weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))},
        )
        # TODO: 如果是 use_moe，需要构建 MoE.Config（省略）
        layers.append(TinyMoETransformerBlock.Config(
            attention_norm=RMSNorm.Config(normalized_shape=dim, param_init={"weight": nn.init.ones_}),
            ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init={"weight": nn.init.ones_}),
            attention=make_gqa_config(
                dim=dim, n_heads=n_heads,
                wqkv_param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
                wo_param_init={"weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))},
                inner_attention=inner_attention, mask_type=mask_type,
            ),
            feed_forward=ffn_cfg,
            # moe=... if use_moe else None,
        ))
    return layers

def _debugmodel(attn_backend: str = "sdpa") -> TinyMoEModel.Config:
    dim, n_heads, n_layers = 256, 8, 4
    return TinyMoEModel.Config(
        dim=dim, vocab_size=2048,
        tok_embeddings=Embedding.Config(num_embeddings=2048, embedding_dim=dim),
        norm=RMSNorm.Config(normalized_shape=dim, param_init={"weight": nn.init.ones_}),
        lm_head=Linear.Config(in_features=dim, out_features=2048),
        rope=None,  # 简化示例：实际应配置 RoPE
        layers=_build_layers(n_layers, dim, n_heads, compute_ffn_hidden_dim(dim), attn_backend),
    )

tinymoe_configs = {"debugmodel": _debugmodel}

def model_registry(flavor: str, attn_backend: str = "sdpa", converters=None):
    config = tinymoe_configs[flavor](attn_backend)
    return ModelSpec(
        name="tiny_moe", flavor=flavor, model=config,
        parallelize_fn=parallelize_tinymoe,  # 见下方
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
```

### 7.3 `parallelize.py`

```python
from torchtitan.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def parallelize_tinymoe(model, *, parallel_dims, training, parallelism,
                        compile_config, ac_config, dump_folder):
    # MVP：单卡/纯 FSDP，什么都不做或只做 FSDP
    # 如果要 TP，在这里加入 model.parallelize(parallel_dims)
    # 如果要 FSDP：
    #   for block in model.layers.values():
    #       fully_shard(block, mesh=dp_mesh, mp_policy=MixedPrecisionPolicy(...))
    #   fully_shard(model, mesh=dp_mesh, mp_policy=...)
    return model
```

### 7.4 `config_registry.py`

```python
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.trainer import Trainer
from . import model_registry

def tinymoe_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",  # 复用 llama3 测试 tokenizer
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=1e-3),
        training=TrainingConfig(local_batch_size=4, seq_len=512, steps=5),
    )
```

### 7.5 `models/__init__.py`

```python
_supported_models = frozenset(["deepseek_v3", "flux", "gpt_oss", "llama3", "llama4",
                               "qwen3", "qwen3_vl", "tiny_moe"])
```

---

## 8. 常见问题速查

**Q: 我能在 `model.py` 里直接用 `nn.Linear` 而不继承 `Module` 吗？**

> 不能。所有子模块必须是 `Module`（或 `Module.from_nn_module` 包装后的类），否则 `model.parallelize()` 递归时会跳过它，导致 TP/SP 的 DTensor 流断裂，最终出现 `plain Tensor + DTensor` 的混合计算错误。`Linear`、`Embedding`、`RMSNorm` 已经帮你做好了钻石继承，直接用即可。

**Q: `sharding.py` 里所有 placement 都要写全 `dp_replicate, dp_shard, cp, tp` 吗？**

> 仅在 `full_dtensor=True` 时需要写全。legacy 路径（默认）下，`resolve_mesh()` 只会保留 `tp` 和 `ep` 轴，其余被过滤掉。但为了代码向前兼容，官方实现总是写全四个 dense 轴。

**Q: 为什么 `FeedForward` 的 `w1, w2, w3` 都是 `Linear.Config`，但 `FeedForward.forward` 里用的是 `self.w1(x)` 而不是 `self.w1.forward(x)`？**

> 因为 `Linear` 继承自 `nn.Linear`，`nn.Linear` 的 `__call__` 会触发 FSDP 的 `fully_shard` 钩子（如果已被 FSDP 包裹）。直接调用 `self.w1(x)` 才能保证这些钩子被触发。这是 PyTorch 的标准行为，不是 torchtitan 特有的。

**Q: 我想加一个自定义层（如新的位置编码），需要改动哪些文件？**

> 1. 如果层本身够通用，先放进 `models/common/` 并继承 `Module`。
> 2. 在 `model.py` 的 `Config` 中新增字段，在 `__init__` 中 `config.my_layer.build()`。
> 3. 如果该层有参数需要 TP 切分，在 `sharding.py` 中为它的 Config 设置 `sharding_config`。
> 4. 如果该层需要 `local_map`（即内部 kernel 不支持 DTensor），参考 `set_gqa_inner_attention_local_map` 的写法设置 `LocalMapConfig`。

---

*本文档基于 torchtitan 当前主干代码编写。随着框架演进，部分 API 可能会发生变化，建议对照实际源码的最新版本进行验证。*
