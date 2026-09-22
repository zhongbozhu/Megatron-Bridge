# Qwen3.5 35B-A3B Text-Only Long-Context SFT on GB200

The BF16 recipe
`qwen35_text_35b_a3b_sft_long_context_32gpu_gb200_bf16_config` targets
32 GB200 GPUs with 131,072-token offline-packed text sequences.

| Setting | Value |
| --- | --- |
| Attention parallelism | TP1, CP8, PP1, DP4 |
| Expert parallelism | EP32, ETP1, EDP1 |
| Global / micro batch size | 32 / 1 |
| Recompute | Selective: `gdn_norm_out` and `moe` |
| Communication | HybridEP; pad uneven dispatch inputs |
| Precision | BF16 compute and FP32 gradient reduction |
| Supervision | Assistant-token SFT; one MTP layer |

Only the language model is constructed. The recipe reads the nested text
configuration from the Qwen3.5 HF model, or accepts an extracted causal-LM
configuration through its `hf_path` argument. Supply a compatible converted
**text-only** pretrained checkpoint, not an unchanged VL checkpoint.

Use `scripts/training/run_recipe.py` with this recipe, `--mode sft`,
and `--step-func gpt_step`. Set `checkpoint.pretrained_checkpoint` and
`dataset.dataset_root` to your local paths. The default relative data path
is only a placeholder, not a request to download a particular dataset.

The dataset uses `GPTSFTDatasetConfig` and `ChatSFTPreprocessingConfig`
with assistant-only supervision. Provide `training.jsonl` and
`validation.jsonl` conversations and let Bridge prepare offline packs, or
set `dataset.offline_packing_specs.packed_train_data_path` and
`packed_val_data_path` to already prepared packs. Keep tokenizer, chat
template, loss masking and packed sequence size consistent with training.
Physical segments use 16-token alignment (2 * CP); padding is not a target.
This is not online or in-batch packing.

For supported HF sources, configure `dataset.hf_dataset` and optionally
`dataset.hf_validation_proportion` to derive a deterministic holdout.
The materializer splits sample indices instead of converting nested
chat/tool payloads to Arrow. This preserves heterogeneous JSON values and
allows a missing split to be regenerated with the same seed.

The MTP objective requires `gpt_step` to propagate the aligned SFT loss mask
to the model (tracked separately in
[PR #6175](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6175)).
That bug fix is not duplicated here. This recipe does not add a vision
encoder, grouped tensors, low-precision compute, or CUDA graphs.
