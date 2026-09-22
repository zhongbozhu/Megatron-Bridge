# Qwen3.5-VL 35B-A3B Long-Context SFT on GB200

The BF16 recipe
`qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config` targets
32 GB200 GPUs with a 131,072-token context.

| Setting | Value |
| --- | --- |
| Attention parallelism | TP2, CP8, PP1, DP2 |
| Expert parallelism | EP32, ETP1, EDP1 |
| Global / micro batch size | 32 / 1 |
| Vision recomputation | Full, one native vision transformer layer per checkpoint |
| Decoder recomputation | Selective: `gdn_norm_out` and `moe`, not core attention |
| Communication | HybridEP; pad uneven dispatch inputs |
| Precision | BF16 compute and FP32 gradient reduction |

The provider's `vision_full_recompute` flag defaults to false, preserving
existing recipes. This recipe enables it independently of decoder
recomputation. It applies to the native vision encoder, not the optional
Hugging Face vision implementation.

Use `scripts/training/run_recipe.py` with this recipe, `--mode sft`,
and `--step-func vlm_step`. Supply your converted VL checkpoint via
`checkpoint.pretrained_checkpoint` and configure the dataset before launch.
The inherited Direct-HF dataset enables step-time in-batch packing with
16-token alignment (2 * CP). Micro batch size one does not by itself combine
multiple source examples. For native buffered multimodal packing, select
`--dataset energon` and provide a prepared Energon dataset and the Qwen
task encoder, including appropriate image/token limits.

The 128K limit includes both text and visual tokens; it does not bound
decoded-image host memory or the replicated vision encoder's memory.
Keep validation data held out and image/packing budgets fixed when comparing
runs. This recipe does not enable grouped tensors, low-precision compute,
forced load balancing, or CUDA graphs.
