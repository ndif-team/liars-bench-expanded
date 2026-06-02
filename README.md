# Liars' Bench Expanded

Documentation for the **Liars' Bench Expanded** dataset, hosted on the [Registry of Open Data on AWS](https://registry.opendata.aws/liars-bench-expanded).

This dataset extends [Liars' Bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) (Kretschmar et al., 2025; [arXiv:2511.16035](https://arxiv.org/abs/2511.16035)) with pre-computed residual-stream activation tensors extracted from open-weight language models at inference time, enabling white-box deception detection research without requiring GPU access.

## Overview

Liars' Bench is a benchmark of 72,863 labeled honest and deceptive outputs from large language models, capturing on-policy lies — responses that appear to contradict the model's own internal beliefs as established in control contexts. The expanded dataset pairs each transcript with layer-wise activation tensors extracted via the [National Deep Inference Fabric (NDIF)](https://ndif.us), covering multiple source datasets and model configurations from an annual red-team competition co-organized by Schmidt Sciences, NDIF, and [Cadenza Labs](https://cadenzalabs.org).

## What's in the dataset

For each record, the dataset provides:

- **Residual-stream activations** at the response-token position, one vector per layer (shape: `n_layers × hidden_dim`, dtype `float16`)
- **Transcript metadata** preserving the original Liars' Bench schema plus expansion fields (see below)

Activations are stored as Zarr v3 arrays partitioned by `captured_model` and `scenario`, enabling efficient layer-slice reads directly from S3 for probe training without downloading the full array.

## Data organization

```
s3://[bucket-name]/
├── README.txt
├── metadata/
│   └── captured_model=[model]/
│       └── scenario=[scenario]-[year]/
│           └── split=[split]/
│               └── records.parquet
└── data/
    └── captured_model=[model]/
        └── scenario=[scenario]-[year]/
            └── split=[split]/
                └── activations.zarr/    # shape: (n_records, n_layers, hidden_dim)
```

Scenario names encode the `(generator_model, source_dataset, year)` tuple using Hive-safe slugs, e.g. `qwen-2.5-72b-instruct_wmdp-bio-2026`. The `captured_model` partition identifies the model whose internal activations were recorded — the target of probe training.

### Zarr array details

| Property | Value |
|---|---|
| Shape | `(n_records, n_layers, hidden_dim)` |
| dtype | `float16` |
| Chunk shape | `(1000, 1, 4096)` |
| Compressor | Blosc + LZ4 |
| Activation position | Response token (last generated token) |

The chunk shape `(1000, 1, 4096)` is optimized for the primary probe-training access pattern: fetching all records for a single layer (`array[:, layer_idx, :]`) requires only `ceil(n_records / 1000)` HTTP requests, each ~8 MB.

### Metadata schema (records.parquet)

| Field | Type | Source | Description |
|---|---|---|---|
| `index` | int64 | Liars' Bench | Record index within source config |
| `model` | string | Liars' Bench | Model that generated the response |
| `messages` | list[{content, role}] | Liars' Bench | Full conversation in chat format |
| `deceptive` | bool | Liars' Bench | Whether the response is labeled deceptive |
| `canary` | string | Liars' Bench | Tracking/watermark string |
| `temperature` | float64 | Liars' Bench | Generation temperature (null if not applicable) |
| `meta` | string | Liars' Bench | Scenario-specific metadata JSON (null if not applicable) |
| `dataset` | string | Liars' Bench | Source benchmark name (null if not applicable) |
| `dataset_index` | int64 | Liars' Bench | Index within source benchmark (null if not applicable) |
| `captured_model` | string | Expanded | Model whose activations were extracted |
| `num_layers` | int64 | Expanded | Number of transformer layers in captured model |
| `hidden_size` | int64 | Expanded | Hidden dimension of captured model |
| `n_response_tokens` | int64 | Expanded | Number of response tokens (activation is at the last) |
| `seq_len` | int64 | Expanded | Full prompt + response sequence length |

Fields marked "null if not applicable" are present in only some Liars' Bench source configs; they are null for records from configs that do not include them.

## Accessing the data

Install dependencies:

```bash
pip install zarr s3fs pandas pyarrow
```

Load activations for a single layer:

```python
import zarr, s3fs, pandas as pd

fs = s3fs.S3FileSystem(anon=True)
store = zarr.open(
    "s3://[bucket-name]/data/captured_model=qwen3.5-9b/scenario=qwen-2.5-72b-instruct_wmdp-bio-2026/split=test/activations.zarr",
    storage_options={"anon": True}
)
# shape: (n_records, n_layers, hidden_dim)

# Fetch all records for layer 15 — downloads ~73 chunks in parallel
layer15 = store[:, 15, :]   # shape: (n_records, 4096)

# Load matching labels
records = pd.read_parquet(
    "s3://[bucket-name]/metadata/captured_model=qwen3.5-9b/scenario=qwen-2.5-72b-instruct_wmdp-bio-2026/split=test/records.parquet",
    filesystem=fs
)

labels = records["deceptive"].values
```

## Source datasets

The initial release covers the following Liars' Bench source configs:

| Config | Description | Records |
|---|---|---|
| `alpaca` | Control dataset — honest and deceptive responses on Alpaca instructions | 8,000 |
| `convincing-game` | Deception in persuasion tasks | 888 |
| `gender-secret` | Identity concealment scenarios | 762 |
| `harm-pressure-choice` | Deception under pressure with harmful choices | 5,400 |
| `harm-pressure-knowledge-report` | Knowledge reporting under pressure | 7,068 |
| `insider-trading` | Financial deception scenarios | 6,873 |
| `instructed-deception` | Directly instructed deceptive responses | 26,426 |
| `soft-trigger` | Steganographic/trigger-based deception | 24,000 |

New model organisms and scenarios from subsequent competition cycles will be added as additional `scenario=` partitions in future releases.

## Citation

If you use this dataset, please cite both the expanded dataset and the original Liars' Bench:

```
Liars' Bench Expanded was accessed on [DATE] at registry.opendata.aws/liars-bench-expanded.
```

```bibtex
@misc{kretschmar2025liarsbench,
  title={Liars' Bench: A Benchmark for Evaluating LLM Deception},
  author={Kretschmar et al.},
  year={2025},
  eprint={2511.16035},
  archivePrefix={arXiv}
}
```

## Contact

- Questions about the dataset: open an issue on this repository or post on [AWS re:Post](https://repost.aws/tags/TApd0Wl5P8S9O6riTWmh-cGw/aws-open-data) using the AWS Open Data tag
- Email: info@ndif.us or info@cadenzalabs.org (subject: "Liars' Bench Expanded")
- Activations extracted via [NDIF](https://ndif.us) using [NNsight](https://nnsight.net)

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
