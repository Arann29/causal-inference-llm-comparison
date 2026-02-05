# Causal Inference Explorer

A Python tool for comparing LLM-based causal reasoning against data-driven bivariate causal discovery algorithms (ROCHE  LOCI, LCUBE) on data with regime-dependent causal directions.

## Prerequisites

- **Docker** and **Docker Compose** installed
- **OpenAI API key** (Optional)

## Quick Start

1. **Configure your API key:**
   ```bash
   cp .env.template .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open your browser to [http://localhost:8501](http://localhost:8501)

## Features

- **Dataset Selection**: Choose from 5 bivariate pairs with literature-based ground truth
- **Visualization**: Scatter plots showing regime-dependent relationships
- **Causal Analysis**: Run ROCHE, LOCI, and LCUBE algorithms per regime
- **LLM Analysis**: Query GPT models using a 4-step prompting chain
- **Batch Experiments**: Run all datasets across all configurations
- **Results Comparison**: Compare algorithm vs LLM predictions against ground truth

## Datasets

| Dataset | Domain | Threshold Variable |
|---------|--------|-------------------|
| Auto MPG-Horsepower | Automotive | horsepower @ 140 |
| Emissions-Renewables | Environment | renewable_share @ 15% |
| UK Interest Rates | Economics | short_rate @ 3.5% |
| US Inflation-Fed Rate | Economics | inflation @ 5.4% |
| Synthetic Stress-Fatigue | Validation | stress @ 0.5 |

## Configuration Options

The tool supports 4 experimental configurations:
- `raw_named_hint_explicit`: Raw data with variable names
- `raw_anon_hint_explicit`: Raw data with anonymized variables
- `segmented_named_hint_explicit`: Pre-segmented data with variable names
- `segmented_anon_hint_explicit`: Pre-segmented data with anonymized variables

## Project Structure

```
├── app/                  # Streamlit application
│   ├── components/       # UI panels
│   └── utils/            # Core utilities (prompts, LLM client, etc.)
├── DATA/                 # Datasets and ground truth
├── LOCI/                 # LOCI algorithm implementation
├── ROCHE/                # ROCHE algorithm implementation
├── LCUBE/                # LCUBE algorithm implementation
├── helpers/              # Shared utilities
└── results/              # Output directory for experiments
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `OPENAI_DAILY_LIMIT` | No | Daily spending limit in USD (default: 5.0) |

## Source Links for Data-Driven Causal Methods

This tool uses for comparison three state-of-the-art causal discovery algorithms for additive noise models:

- **LOCI** (Location-Scale Noise Models) - [Immer et al., 2023](https://github.com/AlexImmer/loci)
- **ROCHE** (Robust Estimation with Student's t-distribution) - [Tran et al., 2023](https://github.com/quangdzuytran/ROCHE)
- **LCUBE** (MDL-based approach with cubic splines) - [Hlavackova-Schindler & Marsela, 2025](https://github.com/suzi216/LCUBE)

Please refer to their original repositories for implementation details and cite their work if you use these algorithms.

## License

This tool accompanies the paper submission for academic review.
