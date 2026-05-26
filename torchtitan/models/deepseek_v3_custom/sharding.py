# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def set_deepseek_v3_custom_sharding_config(*args, **kwargs) -> None:
    """
    No-op placeholder.

    deepseek_v3_custom does not use TP/SP/EP sharding.
    All parallelism is handled via FSDP/HSDP/DDP in parallelize.py.
    """
    pass
