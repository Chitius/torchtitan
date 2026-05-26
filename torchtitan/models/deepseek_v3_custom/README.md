## Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
```

```bash
# DeepSeek 16B tokenizer:
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

> **Note:** We are reusing the tokenizer from deepseek-ai/deepseek-moe-16b-base to help users test and run the 16B model. This is not the official tokenizer for the DeepSeek-V3-16B model. The DeepSeek-V3 model has a different architecture from the deepseek-moe models (different attention implementation, MoE router implementation, etc.), making it not feasible to load deepseek-moe-16b model weights into DeepSeek-V3-16B.


## Training

```bash
# Quick debug run with small model
MODULE=deepseek_v3 CONFIG=deepseek_v3_debugmodel ./run_train.sh
```

```bash
# 16B parameter model: adapted from older 16B parameter model from https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
MODULE=deepseek_v3 CONFIG=deepseek_v3_16b ./run_train.sh
```

```bash
# 671B parameter model
MODULE=deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh
```


## HuggingFace -> DCP Checkpoint Conversion

We implemented StateDictAdapter to perform HuggingFace safetensor to DCP format conversion. Currently, we only support conversion from HF checkpoints to DCP checkpoints offline (using CPU plain tensor).

Run the offline conversion script:
```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B
```


---

## 关于 `deepseek_v3_custom` 模型

### 模型来源与简化

`deepseek_v3_custom` 是基于 torchtitan 官方 `deepseek_v3` 模型定义的一个**简化定制版本**，旨在为开发者提供一个更易于实验和迭代的 DeepSeek-V3 架构实现。主要做了以下简化：

1. **并行策略简化**：移除了对 Tensor Parallelism (TP)、Context Parallelism (CP)、Expert Parallelism (EP) 和 Pipeline Parallelism (PP) 的支持，仅保留 FSDP/HSDP/DDP 作为数据并行手段。这使得模型在单节点或多节点上的部署和调试更加直接。

2. **Sharding 配置简化**：去除了原模型中复杂的 per-parameter sharding 配置和 TP/EP mesh 相关的逻辑，降低了理解门槛。

3. **预设配置精简**：`__init__.py` 中去除了原 `deepseek_v3` 的多个预设（如 16B、671B 等），仅保留一个与 `/home/public/liuyichuan/Deepseek3B.yaml` 语义对齐的 **3B 预设**，方便与 Megatron-LM 的配置进行对比和迁移。

4. **保持核心架构**：保留了 DeepSeek-V3 的核心特性，包括 MLA (Multi-head Latent Attention)、MoE (Mixture of Experts) 路由和专家计算、以及共享专家 (Shared Experts) 等。

### MoE 负载均衡辅助损失 (Aux Loss) 的实现与用法

本模型当前支持两种 MoE 负载均衡机制，且二者**互斥**：

1. **Auxiliary-Loss-Free Load Balancing**（`load_balance_coeff`）：通过 `expert_bias` 实现无辅助损失的负载均衡。这是 torchtitan 的原生机制，在优化器 step 前更新 `expert_bias`。

2. **基于 Loss 的负载均衡**（`aux_loss_coeff` / `seq_aux_loss_coeff`）：
   - **Micro-batch level aux loss** (`aux_loss_coeff`)：基于 Switch Transformer 的负载均衡损失。
   - **Sequence-level aux loss** (`seq_aux_loss_coeff`)：基于 DeepSeek-V2/V3 论文中的序列级负载均衡损失。

**当前 `deepseek_v3_custom` 的 3B 预设默认启用了 `seq_aux_loss_coeff=1e-4`**，同时自动将 `load_balance_coeff` 设为 `None`，以避免两种机制冲突。

**注意事项**：
- 基于 Loss 的负载均衡**暂不支持 Pipeline Parallelism (PP)** 和 **Expert Parallelism (EP)**。如果检测到启用了 PP 或 EP，训练会在初始化阶段抛出 `RuntimeError`。
- `global_avg_loss`（主 loss 指标）**包含了 aux loss 的数值**，因为它在 backward 前被直接累加到了主损失上。wandb 中 `loss_metrics/moe_aux_loss` 单独报告 aux loss，方便你观察其量级和对总 loss 的贡献。
