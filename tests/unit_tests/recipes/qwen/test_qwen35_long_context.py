# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the GB200 text-only Qwen3.5 long-context BF16 recipe."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge.data.builders.gpt_sft import GPTSFTDatasetConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider


recipe_module = importlib.import_module("megatron.bridge.recipes.qwen.gb200.qwen35")
recipe_name = "qwen35_text_35b_a3b_sft_long_context_32gpu_gb200_bf16_config"
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("nested_text_config", [False, True])
@pytest.mark.parametrize("seq_length", [32768, 131072])
def test_text_only_long_context_defaults(monkeypatch, nested_text_config, seq_length):
    text_config = SimpleNamespace()
    hf_config = SimpleNamespace(text_config=text_config) if nested_text_config else text_config
    auto_config = Mock()
    auto_config.from_pretrained.return_value = hf_config
    auto_bridge = Mock()
    provider = GPTModelProvider(num_layers=2, hidden_size=128, num_attention_heads=4)
    auto_bridge.from_hf_config.return_value.to_megatron_provider.return_value = provider
    monkeypatch.setattr(recipe_module, "AutoConfig", auto_config)
    monkeypatch.setattr(recipe_module, "AutoBridge", auto_bridge)

    recipe = getattr(recipe_module, recipe_name)
    cfg = recipe(seq_length=seq_length, hf_path="local/qwen35-text")

    auto_config.from_pretrained.assert_called_once_with("local/qwen35-text")
    auto_bridge.from_hf_config.assert_called_once_with(text_config)
    auto_bridge.from_hf_config.return_value.to_megatron_provider.assert_called_once_with(load_weights=False)
    assert text_config.architectures == ["Qwen3_5MoeForCausalLM"]
    assert cfg.tokenizer.tokenizer_model == "local/qwen35-text"
    assert cfg.model is provider
    assert getattr(importlib.import_module("megatron.bridge.recipes.qwen"), recipe_name) is recipe
    assert getattr(importlib.import_module("megatron.bridge.recipes.qwen.gb200"), recipe_name) is recipe
    assert cfg.model.seq_length == cfg.dataset.seq_length == seq_length
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 8
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.sequence_parallel is False
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["gdn_norm_out", "moe"]
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_hybridep_pad_uneven_dispatch_inputs is True
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.moe_use_grouped_tensor is False
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp8 is None
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert isinstance(cfg.dataset, GPTSFTDatasetConfig)
    assert cfg.dataset.enable_offline_packing is True
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == seq_length
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 16
    assert cfg.dataset.preprocessing.loss_mode == "assistant"
    assert cfg.dataset.do_validation is True
    assert cfg.dataset.do_test is False
