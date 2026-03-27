# Running on La Ruche (Mesocentre Paris-Saclay)

## Step-by-step guide

### 1. Clone the `elora-ruche` branch on La Ruche

```bash
ssh drouilheel@ruche.mesocentre.universite-paris-saclay.fr
cd $WORKDIR
git clone --branch elora-ruche --single-branch --depth 1 https://github.com/helloelora/rl-quantum-circuit-routing.git
cd rl-quantum-circuit-routing
```

This clones only the lightweight `elora-ruche` branch (no results, notebooks, or docs).

### 2. Run the one-time setup

```bash
bash ruche/setup_ruche.sh
```

This creates a conda env `rl_qrouting` in `$WORKDIR/.conda` (to avoid the 50GB `$HOME` quota).

### 3. Submit the training job

```bash
sbatch ruche/train_ruche.sh
```

### 4. Monitor your job

```bash
# Check job status
squeue -u drouilheel

# Watch live output
tail -f rl_qrouting.o<JOB_ID>

# Check job efficiency after completion
seff <JOB_ID>
```

### 5. Retrieve results

Results are saved in `$WORKDIR/rl_qrouting_runs/runs/<run_name>/`.

To copy results to your local machine:
```bash
# From your local machine:
scp -r drouilheel@ruche.mesocentre.universite-paris-saclay.fr:/gpfs/workdir/drouilheel/rl_qrouting_runs/runs/ ./ruche_results/
```

## Training configuration

The default `train_ruche.sh` runs a **3-stage curriculum** (best config from Grid-v5):

| Stage | Topologies | Steps | Depth |
|-------|-----------|-------|-------|
| 1 | linear_5 | 100k | 10 |
| 2 | linear_5 + grid_3x3 | 400k | 12 |
| 3 | grid_3x3 + heavy_hex_19 | 600k | 16 |

Key differences from Colab runs:
- **Stage 2 doubled** to 400k (was 200k) — grid metrics were still improving at cutoff
- **Stage 3 drops linear_5** — already solved, focus compute on hard topologies
- **Stage 3 depth 16** (not 20) — slightly easier to bootstrap heavy_hex learning
- 10 CPU cores, 24GB RAM, 4h walltime on `cpu_med` partition

## SLURM partitions cheat sheet

| Partition | Max time | Use case |
|-----------|----------|----------|
| `cpu_short` | 1h | Quick test runs |
| `cpu_med` | 4h | Standard training |
| `cpu_long` | 168h (7d) | Very long runs |

To change partition/time, edit the `#SBATCH` lines in `train_ruche.sh`.

## Useful commands

```bash
# Cancel a job
scancel <JOB_ID>

# Check your disk quota
ruche-quota

# Interactive session for debugging
srun --nodes=1 --time=01:00:00 -p cpu_short --mem=4G --cpus-per-task=2 --pty /bin/bash
module load anaconda3/2020.02/gcc-9.2.0
source activate rl_qrouting
cd $WORKDIR/rl-quantum-circuit-routing
python -m src.main --help
```
