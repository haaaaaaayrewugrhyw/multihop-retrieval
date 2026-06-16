# Cross-Layer Attention: does *forcing* the feature hierarchy help?

## The idea (the hypothesis under test)

In a CNN the feature hierarchy — simple edges early, complex parts late —
**emerges on its own** during training. This experiment asks: if we instead
**force** that structure by wiring attention along the depth axis (each block
routes into the next via Q/K/V), does the network learn better, faster, or more
sample-efficiently than leaving the hierarchy implicit?

This mechanism is *not new* (see Layer Attention 2019, RLA 2023, RealFormer
2020, Capsule routing 2017). What is under-explored is the **scientific
question** below — not "does accuracy go up 0.5% on ImageNet" but "when, if
ever, does forcing the hierarchy buy you something."

## What we measure (the real headline)

Accuracy vs. **train-set size** ({1%, 5%, 10%, 25%, 100%} of the data).

- **Hypothesis:** attention variants beat the controls *most* at small data, and
  the gap shrinks toward zero as data grows (forcing structure ≈ a prior that
  matters most when data is scarce).
- **Null result is still a result:** if all five curves overlap, forcing the
  hierarchy adds nothing — worth knowing.

## The five models (one shared backbone, fair comparison)

| variant   | cross-layer mechanism                              | role |
|-----------|----------------------------------------------------|------|
| `baseline`| none — plain sequential CNN                         | control (implicit hierarchy) |
| `skip`    | add resized 1x1-projected previous block (no attn) | **control: connection without attention** |
| `A`       | cross-depth **channel** attention                  | Option A |
| `B`       | **layer** attention over all previous block summaries | Option B |
| `C`       | **spatial** cross-layer attention                  | Option C |

`skip` is the control that lets us attribute any win to *attention* rather than
to merely adding cross-layer connections / parameters. `analyze.py` prints param
counts; attention variants are bigger, so if one wins we re-check against a
width-matched baseline.

## Files

- `models.py`  — backbone + the 4 mixers + `HierNet`.
- `train.py`   — training/eval harness; single run, `--quick`, or full `--sweep`.
- `analyze.py` — sample-efficiency table + gap-vs-baseline.

## Run it

**Laptop first signal (small tiers, 2 seeds — minutes):**
```bash
cd experiments/cross_layer_attention
python train.py --quick --dataset mnist
python analyze.py --dataset mnist
```

**Single run:**
```bash
python train.py --dataset mnist --variant B --train-size 1000 --seed 0 --epochs 15
```

**Full sweep (heavy — run on Kaggle):**
```bash
python train.py --sweep --dataset mnist
python train.py --sweep --dataset cifar10
```

## Kaggle (free 16GB GPU — for the full sweep)

New Kaggle Notebook → Settings:
- **Accelerator = GPU T4 x2** (or P100)
- **Internet = ON**  (torchvision downloads MNIST/CIFAR on first run)

Paste into one cell and run (repo is public, no token needed):

```python
!git clone -b cross-layer-attention https://github.com/haaaaaaayrewugrhyw/multihop-retrieval.git
%cd multihop-retrieval/experiments/cross_layer_attention
!python train.py --sweep --dataset mnist
!python train.py --sweep --dataset cifar10
!python analyze.py --dataset mnist
!python analyze.py --dataset cifar10
```

The full sweep is 5 variants x 5 sizes x 3 seeds = 75 runs per dataset.
CIFAR-10 full-data tier dominates the runtime (~2-3h on a T4). If a session
is tight, drop to 2 seeds by editing `run_sweep(... seeds=(0,1))`, or run the
two datasets in separate sessions.

Read results from the printed `analyze.py` table, or download
`results/<dataset>_results.jsonl` from the notebook's Output tab.
