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

## Prepare the text-only checkpoint

Constructing the model from `text_config` does not extract its weights.
Use the following explicit preparation chain before launching training:

```text
Local unquantized Qwen3.5 MoE VL HF snapshot
  -> extract_qwen35_text_checkpoint.py (decoder + LM head + pretrained MTP)
  -> convert.sh import (text-only native Megatron checkpoint)
  -> train.sh (text-only SFT)
```

Run extraction once in a Bridge environment, from the repository root. Reuse
an existing local snapshot of `Qwen/Qwen3.5-35B-A3B` at revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`; the extractor does not download
weights or access the network. Set these example paths to your own locations:

```bash
VL_HF=/data/checkpoints/qwen35-vl-hf
TEXT_HF=/data/checkpoints/qwen35-text-hf
TEXT_MEGATRON=/data/checkpoints/qwen35-text-megatron

uv run python scripts/conversion/extract_qwen35_text_checkpoint.py \
  --source "${VL_HF}" --output "${TEXT_HF}"
```

The extractor removes `model.visual.*`, renames `model.language_model.*` to
`model.*`, and preserves `lm_head.weight` and all `mtp.*` tensors without
changing their values or dtypes. It writes a causal-LM config with one MTP
layer, a new safetensors index, tokenizer/chat-template assets, and
`text-extraction.json`. It does not instantiate a Transformers model, which
could omit auxiliary MTP weights from its state dict.

Only unquantized Qwen3.5 MoE snapshots with one pretrained MTP layer are
supported. Unknown tensor prefixes, missing required tensors, incompatible
configs, or inconsistent shard indexes fail rather than silently dropping
weights. This is not a generic VLM extractor or a native Megatron checkpoint
converter. The source remains unchanged. The output must be a new directory
outside the source. Allow disk space for another copy of the text weights;
extraction uses CPU tensors one input shard at a time. Failed copies remain
in a logged `.incomplete-*` staging directory, not a valid output checkpoint.

Next, import the **extracted** checkpoint. This local example uses four GPUs
with EP4, independently of the training topology; use an appropriate allocation
and the Slurm options in the [conversion guide](../../../scripts/conversion/README.md)
for a distributed conversion job. Do not wrap `convert.sh` in `torchrun`.

```bash
bash scripts/conversion/convert.sh import \
  --executor local --device gpu --gpus-per-node 4 \
  --hf-model "${TEXT_HF}" --megatron-path "${TEXT_MEGATRON}" \
  --torch-dtype bfloat16 --tp 1 --pp 1 --ep 4 --etp 1
```

Conversion and extraction are weight preparation, not training-state resume:
they do not preserve optimizer, RNG, or dataloader state from a previous run.

## Prepare offline-packed data

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

Use the extracted tokenizer for packing and training. For already materialized
chat JSONL, the existing standalone preparation entry point can generate the
packed artifacts before reserving the training GPUs:

```bash
RECIPE=qwen35_text_35b_a3b_sft_long_context_32gpu_gb200_bf16_config
DATA_ROOT=/data/coderforge/materialized
PACKED_ROOT=/data/coderforge/packed-128k

uv run python scripts/training/prepare_gpt_sft_packed_data.py \
  --recipe "${RECIPE}" --hf-path "${TEXT_HF}" \
  --train-input-path "${DATA_ROOT}/training.jsonl" \
  --val-input-path "${DATA_ROOT}/validation.jsonl" \
  --packed-train-data-path "${PACKED_ROOT}/training.parquet" \
  --packed-val-data-path "${PACKED_ROOT}/validation.parquet" \
  --packed-metadata-path "${PACKED_ROOT}/metadata.jsonl"
```

## Launch with the unified entry point

Use `scripts/training/train.sh` with the recipe, `--mode sft`, and
`--step-func gpt_step`. The example below uses shared paths visible on every
node. Supply your own account, partition, image and output directory. Mount
the checkout at `/opt/Megatron-Bridge` so the job uses these scripts and its
`3rdparty/Megatron-LM`, rather than an older container installation.

```bash
bash scripts/training/train.sh \
  --nodes 8 --gpus-per-node 4 \
  --account "${SLURM_ACCOUNT}" --partition "${SLURM_PARTITION}" \
  --container-image "${CONTAINER_IMAGE}" \
  --mount "${PWD}:/opt/Megatron-Bridge" --mount /data \
  --recipe "${RECIPE}" --mode sft --step-func gpt_step \
  checkpoint.pretrained_checkpoint="${TEXT_MEGATRON}" \
  checkpoint.hf_source_path="${TEXT_HF}" \
  checkpoint.load=null checkpoint.save=/data/runs/qwen35-text-sft \
  tokenizer.tokenizer_model="${TEXT_HF}" \
  dataset.dataset_root="${DATA_ROOT}" \
  dataset.offline_packing_specs.packed_train_data_path="${PACKED_ROOT}/training.parquet" \
  dataset.offline_packing_specs.packed_val_data_path="${PACKED_ROOT}/validation.parquet" \
  dataset.offline_packing_specs.packed_metadata_path="${PACKED_ROOT}/metadata.jsonl"
```

The recipe still resolves the pinned original HF architecture config when it
is constructed. For offline runs, cache that config in the container's
`HF_HOME` beforehand and forward/mount the cache using the training launcher.
The extracted directory supplies tokenizer assets and the checkpoint source
used here. `dataset.dataset_root` is only a local-data placeholder by default,
not a request to download CoderForge. Do not add `--dataset coderforge` to this
prepacked example: dataset selection replaces the recipe's dataset config.

The MTP objective requires `gpt_step` to propagate the aligned SFT loss mask
to the model (tracked separately in
[PR #6175](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6175)).
That bug fix is not duplicated here. This recipe does not add a vision
encoder, grouped tensors, low-precision compute, or CUDA graphs.
