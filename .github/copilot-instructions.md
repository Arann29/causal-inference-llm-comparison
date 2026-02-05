# Copilot Instructions for Causal Inference Explorer

## 🎯 Repository Context
This is a **clean, production-ready distribution** of the Causal Inference Explorer - a standalone application ready for deployment and extension. The codebase is organized for:
- Easy Docker deployment (recommended)
- Clear separation of concerns (UI, algorithms, utilities)
- Extensibility (add datasets, algorithms, LLM configurations)
- Reproducible experiments with ground truth evaluation

This is NOT a research/development workspace - it's the minimal setup for users to run and customize the tool.

## Project Overview
Evaluates **LLM-based causal reasoning** vs **data-driven algorithms** (ROCHE, LOCI, LCUBE) for **regime-dependent bivariate causal discovery**. Key innovation: detecting when causal direction switches (X→Y ↔ Y→X) across data regimes based on literature-backed thresholds.

## Architecture & Data Flow

### Core Pipeline
1. **Dataset Loading** ([DATA/custom_pairs/](../DATA/custom_pairs/)) → 2. **Segmentation** (threshold-based) → 3. **Causal Analysis** (per-regime) → 4. **Evaluation** (vs ground truth)

### Critical Components
- **Ground Truth**: [DATA/pairmeta_with_ground_truth.txt](../DATA/pairmeta_with_ground_truth.txt) defines thresholds and expected causal directions per regime
  - Format: `dataset|cause_cols|effect_cols|weight|threshold_var|threshold_val|regime_low_dir|regime_high_dir`
  - Directions: `1` = X→Y, `-1` = Y→X (numeric convention used throughout)
- **Dynamic Algorithm Loading**: [helpers/causal_param_optimization.py](../helpers/causal_param_optimization.py) imports LOCI/ROCHE/LCUBE via `importlib` to avoid path conflicts
- **4-Step LLM Prompting**: [app/utils/prompts.py](../app/utils/prompts.py) implements sequential reasoning chain to prevent anchoring bias
- **Experiment Configs**: [app/utils/llm_config.py](../app/utils/llm_config.py) defines 4 active presets (raw/segmented × named/anonymous)

### Session State Management (Streamlit)
Key states in [app/main.py](../app/main.py):
- `selected_dataset`: Currently loaded dataset metadata
- `current_dataframe`: Loaded pandas DataFrame
- `segmentation_result`: Tuple of (segments_dict, metadata)
- `causal_results_by_method`: Dict storing {method_name: [(segment_id, direction, score)]}
- `llm_conclusion`: Final JSON response from LLM analysis

## Critical Conventions

### Causal Direction Encoding
**Always use numeric convention** in code:
- `direction = 1`: X causes Y (X→Y)
- `direction = -1`: Y causes X (Y→X)
- `direction = 0`: Uncertain/no clear direction

String representations (for display only): "X->Y" or "Variable_X->Variable_Y"

### Ground Truth Lookup
```python
from app.utils.ground_truth import get_ground_truth
gt = get_ground_truth(dataset_name)  # Auto-normalizes name
threshold_var = gt['threshold_var']   # e.g., 'horsepower'
threshold_val = gt['threshold_val']   # e.g., 140.0
regime_low_dir = gt['regime_low_dir'] # -1 or 1
regime_high_dir = gt['regime_high_dir']
```

### Segmentation Strategy
Literature-based thresholds (NOT data-driven clustering):
```python
from helpers.visualization_segmentation import segment_data
segments, metadata = segment_data(
    df, method='threshold',
    threshold_variable='horsepower', threshold_value=140.0
)
# Returns: segments={0: df_low, 1: df_high}, metadata={...}
```

### Running Causal Algorithms
```python
from helpers.causal_param_optimization import run_segmented_causal_switch
results = run_segmented_causal_switch(
    df=df, method='roche',  # or 'loci', 'lcube'
    segmentation_strategy='threshold',
    segmentation_kwargs={'threshold_variable': var, 'threshold_value': val},
    device='cpu', cause_col=col1, effect_col=col2
)
# Returns: [(segment_id, direction, score), ...]
```

## Development Workflows

### Docker Setup (Primary Method - Recommended for Users)
```bash
# 1. Setup environment
cp .env.template .env
# Edit .env: Add OPENAI_API_KEY=sk-...

# 2. Build and run (Streamlit on :8501, FastAPI on :8000)
docker-compose up --build

# 3. Access UI
http://localhost:8501

# Clean shutdown
docker-compose down
```

This is the recommended deployment method - everything runs in containers with consistent dependencies.

### Local Development (Without Docker)
```bash
# Requires: Python 3.10+, pip
pip install -r requirements.txt

# Set environment
$env:PYTHONPATH="$PWD"
$env:OPENAI_API_KEY="sk-..."

# Run Streamlit
streamlit run app/main.py

# Run FastAPI (if needed)
uvicorn app.api.endpoints:app --reload --port 8000
```

### Running Batch Experiments
Navigate to **"🚀 Batch Run"** tab in UI:
1. Select datasets (default: all 5)
2. Select configurations (default: all 4 presets)
3. Enable LLM analysis (optional, costs API credits)
4. Click "Run Batch" → saves to `results/batch_results_TIMESTAMP.csv` + `prompt_logs/`

### Testing Ground Truth Parsing
```bash
cd causal_inference_app_v1
python -c "from app.utils.ground_truth import parse_pairmeta; print(parse_pairmeta())"
```

## Key File Patterns

### Adding New Datasets (User Extension Point)
1. Create `DATA/custom_pairs/my_dataset.txt` (tab-delimited, no headers, 2+ numeric columns)
2. Add ground truth to `DATA/pairmeta_with_ground_truth.txt`:
   ```
   my_dataset|1 1|2 2|1|threshold_var|threshold_val|low_dir|high_dir
   ```
3. Restart app - Docker auto-detects via volume mount (or refresh browser for local dev)

**Example**: See existing datasets in `DATA/custom_pairs/` as templates (auto_mpg_horsepower.txt, interest_inflation.txt, etc.)

### Modifying LLM Prompts
Edit [app/utils/prompts.py](../app/utils/prompts.py) - chain steps:
- **Step 1**: Domain understanding (competing hypotheses)
- **Step 2**: Regime identification (threshold discovery)
- **Step 3**: Causal analysis (per-regime directions)
- **Step 4**: JSON extraction (structured output)

**Critical**: Step 2 uses `_build_step2_explicit()` to dynamically inject threshold hints based on `ExperimentConfig.context_hint_level`

### Evaluation Against Ground Truth
```python
from app.utils.evaluation import calculate_regime_accuracy
accuracy = calculate_regime_accuracy(
    predictions={'regime_low': -1, 'regime_high': 1},
    ground_truth=gt
)
# Returns: {regime_low_correct, regime_high_correct, overall_accuracy}
```

## Project-Specific Quirks

1. **Algorithm Path Resolution**: LOCI/ROCHE/LCUBE use `causa/__init__.py` structure. Always import via [helpers/causal_param_optimization.py](../helpers/causal_param_optimization.py) dynamic loader, not direct imports.

2. **Streamlit Component Structure**: Each panel ([app/components/](../app/components/)) has `render()` function called from [app/main.py](../app/main.py) tabs. Session state synchronization critical.

3. **LLM Config System**: [app/utils/llm_config.py](../app/utils/llm_config.py) defines `ExperimentConfig` dataclass with 4 active presets. Current focus: `gpt-5.2` only (Claude/Gemini implemented but not used in experiments).

4. **Results Storage**: [results/](../results/) directory has `causal/`, `clustering/`, `llm_responses/` subdirs. Batch runs create timestamped folders with per-dataset CSV + prompt logs.

5. **Data Summa & Extension Points

**Add new causal algorithm**: Create `NEW_ALGO/causa/__init__.py` + `NEW_ALGO/causa/new_algo.py`, update [helpers/causal_param_optimization.py](../helpers/causal_param_optimization.py) to handle `method='new_algo'`. Follow LOCI/ROCHE/LCUBE structure.

**Debug segmentation**: Check [helpers/visualization_segmentation.py](../helpers/visualization_segmentation.py) `segment_data()` function. Verify threshold variable exists in dataframe.

**Trace LLM failures**: Batch runs automatically save logs to `results/prompt_logs/{config}/{dataset}/` as ZIP download. Each step's prompt and response saved separately.

**Modify evaluation metrics**: Edit [app/utils/evaluation.py](../app/utils/evaluation.py) - currently uses per-regime binary accuracy + regime-switch detection.

**Customize LLM experiments**: Add new presets in [app/utils/llm_config.py](../app/utils/llm_config.py) `PRESET_CONFIGS` dict. Modify prompt chain in [app/utils/prompts.py](../app/utils/prompts.py).

## Clean Repository Notes

- **No research artifacts**: This repo contains only production code, no experimental notebooks or draft scripts
- **Minimal dependencies**: See [requirements.txt](../requirements.txt) - core ML/viz packages only
- **Standardized structure**: All algorithms follow `causa/` module pattern for consistency
- **Ready-to-deploy**: Docker setup includes all services (Streamlit UI, FastAPI backend, Ollama for local LLMs)set}/step*.txt`.

**Modify evaluation metrics**: Edit [app/utils/evaluation.py](../app/utils/evaluation.py) - currently uses per-regime binary accuracy + regime-switch detection.
