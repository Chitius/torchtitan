# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter


class DeepSeekV3CustomStateDictAdapter(DeepSeekV3StateDictAdapter):
    """
    StateDictAdapter for deepseek_v3_custom.

    Inherits the full HF ↔ DCP mapping from the base DeepSeekV3 adapter.
    The expert weight split/concat logic works correctly under FSDP-only
    parallelism, so we reuse it without modification.
    """
    pass
