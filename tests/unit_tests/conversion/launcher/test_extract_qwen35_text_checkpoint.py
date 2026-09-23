import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/conversion/extract_qwen35_text_checkpoint.py"
spec = importlib.util.spec_from_file_location("extract_qwen35_text_checkpoint_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extraction)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_snapshot(root: Path, *, sharded: bool = True) -> tuple[Path, dict[str, torch.Tensor]]:
    source = root / "vl"
    source.mkdir()
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "tie_word_embeddings": False,
            "eos_token_id": 42,
            "torch_dtype": "bfloat16",
            "vision_config": {"hidden_size": 4},
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "tie_word_embeddings": True,
            },
        },
    )
    _write_json(source / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast"})
    _write_json(source / "tokenizer.json", {"test_vocabulary": True})
    _write_json(source / "generation_config.json", {"eos_token_id": 42})
    (source / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (source / "chat_templates").mkdir()
    (source / "chat_templates/default.jinja").write_text("{{ messages }}", encoding="utf-8")
    _write_json(source / "preprocessor_config.json", {"image_processor_type": "Qwen3VLImageProcessor"})

    decoder = {
        "model.language_model.embed_tokens.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "model.language_model.norm.weight": torch.ones(4, dtype=torch.float32),
        "model.language_model.layers.0.mlp.gate.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "model.visual.merger.weight": torch.ones(2, 4, dtype=torch.bfloat16),
    }
    mtp = {
        "lm_head.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "mtp.fc.weight": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
        "mtp.norm.weight": torch.ones(4, dtype=torch.float32),
        "mtp.pre_fc_norm_embedding.weight": torch.full((4,), 2, dtype=torch.float32),
        "mtp.pre_fc_norm_hidden.weight": torch.full((4,), 3, dtype=torch.float32),
        "mtp.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
    }
    vision = {"model.visual.patch_embed.proj.weight": torch.ones(4, 3, 2, 2, dtype=torch.bfloat16)}
    tensors = {**decoder, **mtp, **vision}
    if sharded:
        weight_map = {}
        for index, shard in enumerate((decoder, mtp, vision), start=1):
            filename = f"model-{index:05d}-of-00003.safetensors"
            save_file(shard, source / filename)
            weight_map.update(dict.fromkeys(shard, filename))
        _write_json(source / "model.safetensors.index.json", {"weight_map": weight_map})
    else:
        save_file(tensors, source / "model.safetensors")
    return source, tensors


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.language_model.layers.0.weight", "model.layers.0.weight"),
        ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("mtp.layers.0.weight", "mtp.layers.0.weight"),
        ("model.visual.patch_embed.weight", None),
    ],
)
def test_text_weight_names(name, expected):
    assert extraction.text_weight_name(name) == expected


def test_unknown_tensor_is_not_silently_discarded():
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        extraction.text_weight_name("unexpected_adapter.weight")


@pytest.mark.parametrize("sharded", [True, False])
def test_extract_preserves_decoder_mtp_and_tokenizer(tmp_path, sharded):
    source, tensors = _make_snapshot(tmp_path, sharded=sharded)
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    output = tmp_path / "text"
    extraction.extract_checkpoint(source=source, output=output)

    expected = {
        extraction.text_weight_name(name): tensor
        for name, tensor in tensors.items()
        if not name.startswith("model.visual.")
    }
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(expected)
    shard_names = set(index["weight_map"].values())
    assert len(shard_names) == (2 if sharded else 1)
    actual = {}
    for name in shard_names:
        shard = load_file(output / name)
        assert set(shard) == {key for key, value in index["weight_map"].items() if value == name}
        actual.update(shard)
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0, check_dtype=True)
    total_size = sum(t.numel() * t.element_size() for t in expected.values())
    assert index["metadata"]["total_size"] == total_size

    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "qwen3_5_moe_text"
    assert config["architectures"] == ["Qwen3_5MoeForCausalLM"]
    assert config["hidden_size"] == 4
    assert config["tie_word_embeddings"] is False
    assert config["eos_token_id"] == 42
    assert config["torch_dtype"] == "bfloat16"
    assert "vision_config" not in config and "text_config" not in config
    for field in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_num_layers"):
        assert config[field] == 1
    manifest = json.loads((output / "text-extraction.json").read_text())
    assert manifest["source"] == str(source.resolve())
    assert manifest["mtp_num_layers"] == 1
    assert manifest["tensor_count"] == len(expected)
    assert manifest["total_size"] == total_size
    for name in ("tokenizer_config.json", "tokenizer.json", "generation_config.json", "chat_template.jinja"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert (output / "chat_templates/default.jinja").read_bytes() == (
        source / "chat_templates/default.jinja"
    ).read_bytes()
    assert not (output / "preprocessor_config.json").exists()
    assert before == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    assert not list(tmp_path.glob(".text.incomplete-*"))


@pytest.mark.parametrize("quantization_location", ["parent", "text"])
def test_quantized_checkpoint_rejected(tmp_path, quantization_location):
    source, _ = _make_snapshot(tmp_path)
    config = json.loads((source / "config.json").read_text())
    target = config if quantization_location == "parent" else config["text_config"]
    target["quantization_config"] = {"quant_method": "fp8"}
    _write_json(source / "config.json", config)
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        extraction.extract_checkpoint(source=source, output=tmp_path / "text")
    assert not (tmp_path / "text").exists()


@pytest.mark.parametrize("defect", ["no-mtp", "two-mtp-layers", "missing-mtp-fc", "wrong-model"])
def test_incompatible_model_or_mtp_rejected(tmp_path, defect):
    source, tensors = _make_snapshot(tmp_path)
    config = json.loads((source / "config.json").read_text())
    names = [extraction.text_weight_name(name) for name in tensors if not name.startswith("model.visual.")]
    if defect == "no-mtp":
        names = [name for name in names if not name.startswith("mtp.")]
    elif defect == "two-mtp-layers":
        names.append("mtp.layers.1.self_attn.q_proj.weight")
    elif defect == "missing-mtp-fc":
        names.remove("mtp.fc.weight")
    else:
        config["text_config"]["model_type"] = "qwen3_5_text"
    with pytest.raises(ValueError):
        extraction.text_model_config(config, names)


@pytest.mark.parametrize("output_kind", ["existing", "inside-source"])
def test_extraction_never_overwrites_source_or_existing_output(tmp_path, output_kind):
    source, _ = _make_snapshot(tmp_path)
    output = source if output_kind == "existing" else source / "text"
    before = (source / "config.json").read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        extraction.extract_checkpoint(source=source, output=output)
    assert (source / "config.json").read_bytes() == before
    assert not (source / "text").exists()


@pytest.mark.parametrize("filename", ["../outside.safetensors", "subdir/weights.safetensors", "C:weights.safetensors"])
def test_weight_index_rejects_paths(tmp_path, filename):
    source, _ = _make_snapshot(tmp_path)
    _write_json(source / "model.safetensors.index.json", {"weight_map": {"lm_head.weight": filename}})
    with pytest.raises(ValueError, match="plain safetensors shard filename"):
        extraction.extract_checkpoint(source=source, output=tmp_path / "text")


def test_corrupt_shard_index_does_not_publish_checkpoint(tmp_path):
    source, _ = _make_snapshot(tmp_path)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    del index["weight_map"]["model.visual.merger.weight"]
    _write_json(source / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="index does not match"):
        extraction.extract_checkpoint(source=source, output=tmp_path / "text")
    assert not (tmp_path / "text").exists()
    assert len(list(tmp_path.glob(".text.incomplete-*"))) == 1


def test_missing_tokenizer_does_not_publish_checkpoint(tmp_path):
    source, _ = _make_snapshot(tmp_path)
    (source / "tokenizer_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer_config"):
        extraction.extract_checkpoint(source=source, output=tmp_path / "text")
    assert not (tmp_path / "text").exists()


def test_write_failure_does_not_publish_partial_checkpoint(tmp_path, monkeypatch):
    source, _ = _make_snapshot(tmp_path)
    writes = 0

    def fail_second_shard(tensors, filename, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated full disk")
        save_file(tensors, filename, **kwargs)

    monkeypatch.setattr("safetensors.torch.save_file", fail_second_shard)
    with pytest.raises(OSError, match="simulated full disk"):
        extraction.extract_checkpoint(source=source, output=tmp_path / "text")
    assert not (tmp_path / "text").exists()
    incomplete = list(tmp_path.glob(".text.incomplete-*"))
    assert len(incomplete) == 1
    assert len(list(incomplete[0].glob("*.safetensors"))) == 1


def test_cli_extracts_local_checkpoint(tmp_path):
    source, _ = _make_snapshot(tmp_path)
    output = tmp_path / "text"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Text checkpoint ready" in result.stderr
    assert json.loads((output / "text-extraction.json").read_text())["mtp_num_layers"] == 1
