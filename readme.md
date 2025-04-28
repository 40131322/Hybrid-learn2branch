
---

# Running the Code

## Step 1: Generate Facilities

```bash
python 06_generate_custom_facilities.py
python 06_generate_custom_facilities2.py
```

## Step 2: Generate Dataset

```bash
python 07_generate_dataset.py facilities
python 07_generate_experiment_dataset.py facilities
```

> **Note:**  
> `facilities` = the problem type (matches your `.pkl` sample dataset)

## Step 3: Train Models

### Train GCNN Baseline

```bash
python 08_train_gnn_torch.py facilities -m baseline_torch -g -1 -s 42
```

- `-m baseline_torch` = model name (matches your `models/baseline_torch/model.py` file)
- `-g -1` = CPU only (`-g 0` for first GPU)
- `-s 42` = random seed

### Train Hybrid Models (FILM)

```bash
# Basic training
python 08_train_hybrid.py facilities -m film

# Train on GPU 0 with seed 42
python 08_train_hybrid.py facilities -m film -g 0 -s 42

# Train with distillation
python 08_train_hybrid.py facilities -m film --distilled

# Train without end-to-end (GCNN frozen)
python 08_train_hybrid.py facilities -m film --no_e2e

# Add auxiliary task (e.g., ED auxiliary loss)
python 08_train_hybrid.py facilities -m film --at ED --beta_at 0.1
```

### Train MLP

```bash
python 08_train_mlp.py facilities -g 0 -s 42 -w sigmoidal_decay
```

## Step 4: Test Models

### Test GCNN Model

```bash
python 09_test_gcnn_torch.py setcover -g 0
```

- `setcover` = problem type (`setcover`, `cauctions`, `facilities`, `indset`)
- `-g 0` = use GPU id 0 (`-g -1` for CPU)

Optional:
```bash
--test_path path/to/your/test_folder
```

### Test Hybrid Model

```bash
python 09_test_hybrid.py setcover -g 0 --model_string film
```

- `--model_string film` = match model folders containing "film" in `trained_models/setcover/`

Alternative: specify exact folder name:
```bash
--model_name exact_model_folder_name
```

### Test MLP Model

```bash
python 09_test_mlp.py setcover -g 0
```

- Automatically picks up any folder inside `trained_models/setcover/` that starts with `mlp_`.

## Step 5: Advanced Testing

### Test GCNN (Custom Config)

```bash
python 10_test_gcnn_torch.py setcover --config 200_50_3 --model_name baseline_torch --gpu 0
```

- `--config 200_50_3` = dataset configuration
- `--model_name baseline_torch` = GCNN model to load

Optional flags:
```bash
--max_samples 100
--start_idx 50
```

### Test Hybrid (Custom Config)

```bash
python 10_test_hybrid.py setcover --config 200_50_3 --model_name hybrid_model --gpu 0
```

- `--model_name hybrid_model` = exact folder name
- or `-m hybrid` for partial match

### Test MLP (Custom Config)

```bash
python 10_test_mlp.py facilities --config 200_100_5 --model_name mlp_model --gpu 0
```

- `--model_name` looks inside `trained_models/facilities/`
- Alternatively specify model path directly using `--model_path`

---

Would you also like me to format it as a ready-to-use `.md` file download? 🚀g