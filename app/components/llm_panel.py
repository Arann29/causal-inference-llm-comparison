import streamlit as st
import pandas as pd
import json
from app.utils.llm_client import get_llm_client
from app.utils.prompts import get_prompt_chain
from app.utils.llm_config import ExperimentConfig, get_preset_config, get_available_presets, AVAILABLE_MODELS
from app.utils.data_summarizer import generate_summary


from typing import Optional, List


def render_config_panel() -> ExperimentConfig:
    """Render configuration UI and return ExperimentConfig."""
    st.subheader("⚙️ Experiment Configuration")
    
    # Model selection
    selected_model = st.selectbox("Model:", AVAILABLE_MODELS, index=0)
    
    # Get all available presets (all are now _hint_explicit variants)
    all_presets = get_available_presets()
    
    # Create display names for the dropdown
    preset_display_names = {p: get_preset_config(p).scenario_name for p in all_presets}
    
    # Preset selection dropdown
    selected_display_name = st.selectbox(
        "Configuration Preset:",
        options=list(preset_display_names.values()),
        index=list(preset_display_names.values()).index('Raw+Stats | Named | Hint | Explicit')  # Default
    )
    
    # Get the actual preset name from display name
    preset_name = [k for k, v in preset_display_names.items() if v == selected_display_name][0]
    
    config = get_preset_config(preset_name)
    
    # Update model from dropdown selection
    config.model_name = selected_model
    
    # Display configuration details
    with st.expander("📋 Configuration Details", expanded=False):
        st.markdown(f"""
        **Active Configuration:** `{preset_name}`
        
        {config.get_summary_description()}
        
        - 🤖 **Model**: {config.model_name}
        - 🌡️ **Temperature**: {config.temperature}
        - 🎲 **Random Seed**: {config.raw_data_options.get('random_seed', 'N/A')}
        """)
    
    # Store in session state
    st.session_state['experiment_config'] = config
    
    return config


def summarize_data(df: pd.DataFrame) -> str:
    """Generate a textual statistical summary for the LLM."""
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."

    x, y = df.columns[:2]
    desc = df[[x, y]].describe().T
    corr = df[[x, y]].corr().iloc[0, 1]

    summary = (
        f"The dataset contains two variables: '{x}' and '{y}'.\n\n"
        f"Variable '{x}':\n"
        f"  - Mean: {desc.loc[x, 'mean']:.2f}, Std Dev: {desc.loc[x, 'std']:.2f}\n"
        f"  - Range: [{desc.loc[x, 'min']:.2f}, {desc.loc[x, 'max']:.2f}]\n\n"
        f"Variable '{y}':\n"
        f"  - Mean: {desc.loc[y, 'mean']:.2f}, Std Dev: {desc.loc[y, 'std']:.2f}\n"
        f"  - Range: [{desc.loc[y, 'min']:.2f}, {desc.loc[y, 'max']:.2f}]\n\n"
        f"The Pearson correlation between '{x}' and '{y}' is {corr:.2f}."
    )
    return summary

def run_prompt_chain(summary, config: Optional[ExperimentConfig] = None, threshold_var: Optional[str] = None, threshold_value: Optional[float] = None, df_columns: Optional[List[str]] = None):
    """Runs the optimized 4-step prompt chain with optional configuration.
    
    The summary is injected into ALL steps to maintain grounding in original data.
    This prevents telephone-game effects where later steps only reason over prior interpretations.
    """
    llm_client = get_llm_client("openai")
    
    # Get model and temperature from config if provided, otherwise use defaults
    # model_name = config.model_name if config else "gpt-4.1"
    model_name = config.model_name if config else "gpt-5.2"
    temperature = config.temperature if config else 0.0
    
    prompt_chain = get_prompt_chain(config, threshold_var, threshold_value, df_columns)
    
    history = ""
    results = []

    for i, step in enumerate(prompt_chain):
        # Inject summary into ALL steps to maintain grounding in original data
        prompt = step["prompt"].format(summary=summary, history=history)
        
        with st.spinner(f"Running {step['name']}..."):
            response = llm_client.ask(prompt, model=model_name, temperature=temperature)
        
        if response:
            results.append({
                "step": step["name"], 
                "response": response,
                "prompt_sent": prompt  # Store the actual prompt for debugging
            })
            history += f"**{step['name']}**: {response}\n\n"
        else:
            st.error(f"Failed to get response for {step['name']}")
            return None
            
    return results

def render():
    st.header("🤖 LLM-Based Causal Reasoning")

    if "current_dataframe" not in st.session_state or st.session_state.current_dataframe is None:
        st.info("Please load a dataset and apply segmentation first.")
        return

    df = st.session_state.current_dataframe
    dataset_name = st.session_state.get('selected_dataset', {}).get('name', '').replace('.txt', '')
    
    # Render configuration panel
    config = render_config_panel()
    
    # Display configuration summary
    st.write("---")
    st.subheader("📋 Current Configuration")
    st.markdown(config.get_summary_description())
    
    st.write("---")
    st.subheader("📊 Dataset Summary for LLM")
    summary = generate_summary(df, config, dataset_name)
    st.code(summary, language="markdown")

    # Add repeated trials configuration
    col1, col2 = st.columns([2, 1])
    with col1:
        run_button = st.button("🚀 Run LLM Analysis", type="primary", use_container_width=True)
    with col2:
        n_trials = st.number_input("Trials", min_value=1, max_value=20, value=1, 
                                   help="Set to 1 for single run (faster, cheaper)\nSet to 10+ for consistency analysis")

    if run_button:
        st.session_state.llm_results = None  # Clear previous results
        st.session_state.llm_all_trials = []  # Store all trials
        st.session_state.llm_conclusion = None
        st.session_state.llm_config = config.to_dict()  # Store configuration
        
        # Get threshold from ground truth if available
        from app.utils.ground_truth import get_ground_truth
        ground_truth = get_ground_truth(dataset_name)
        threshold_var = ground_truth.get('threshold_var') if ground_truth else None
        threshold_value = ground_truth.get('threshold_val') if ground_truth else None
        
        # Progress tracking (only show for multiple trials)
        if n_trials > 1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        all_trial_results = []
        all_trial_conclusions = []
        
        for trial_num in range(n_trials):
            if n_trials > 1:
                status_text.text(f"Running trial {trial_num + 1} of {n_trials}...")
                progress_bar.progress((trial_num) / n_trials)
            
            # Get dataframe columns for anonymization mapping
            df_columns = df.columns[:2].tolist() if df is not None else None
            
            results = run_prompt_chain(summary, config, threshold_var, threshold_value, df_columns)
            
            if results:
                all_trial_results.append(results)
                
                # Try to parse the final JSON conclusion for this trial
                try:
                    final_conclusion_str = results[-1]['response']
                    # Strip markdown code blocks
                    if '```json' in final_conclusion_str:
                        final_conclusion_str = final_conclusion_str.split('```json')[1].split('```')[0]
                    elif '```' in final_conclusion_str:
                        final_conclusion_str = final_conclusion_str.split('```')[1].split('```')[0]
                    
                    final_conclusion_json = json.loads(final_conclusion_str.strip())
                    all_trial_conclusions.append(final_conclusion_json)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    if n_trials > 1:
                        st.warning(f"Trial {trial_num + 1}: Could not parse JSON conclusion: {e}")
                    else:
                        st.warning(f"Could not parse JSON conclusion: {e}")
                    all_trial_conclusions.append(None)
        
        if n_trials > 1:
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed {n_trials} trials")
        
        # Store all trials in session state
        st.session_state.llm_all_trials = all_trial_results
        
        if all_trial_results:
            # Use the first trial as the primary results for display
            st.session_state.llm_results = all_trial_results[0]
            
            # Calculate consistency and majority vote if multiple trials
            if n_trials > 1 and all_trial_conclusions:
                from app.utils.ground_truth import map_prediction_to_numeric
                from app.utils.evaluation import calculate_llm_consistency
                from collections import Counter
                
                # Extract regime-specific predictions from each trial
                trial_predictions = []
                for conclusion in all_trial_conclusions:
                    if conclusion and 'directions' in conclusion:
                        directions = conclusion['directions']
                        if len(directions) >= 2:
                            regime_directions = {
                                d.get('regime', d.get('cluster', idx)): d.get('direction', '')
                                for idx, d in enumerate(directions)
                            }
                            dir_low = regime_directions.get(0, directions[0].get('direction', ''))
                            dir_high = regime_directions.get(1, directions[1].get('direction', ''))
                            pred = {
                                'regime_low': map_prediction_to_numeric(dir_low),
                                'regime_high': map_prediction_to_numeric(dir_high)
                            }
                            trial_predictions.append(pred)
                
                # Calculate consistency metrics
                if trial_predictions:
                    consistency_metrics = calculate_llm_consistency(trial_predictions)
                    st.session_state.llm_consistency = consistency_metrics
                    
                    # Use majority vote as final conclusion
                    majority_conclusion = {
                        'directions': [
                            {
                                'regime': 0,  # Fixed: use 0 for low regime
                                'direction': 'X→Y' if consistency_metrics['regime_low_majority'] == 1 
                                           else ('Y→X' if consistency_metrics['regime_low_majority'] == -1 else 'Uncertain'),
                                'confidence': f"{consistency_metrics['regime_low_consistency']:.0f}%"
                            },
                            {
                                'regime': 1,  # Fixed: use 1 for high regime
                                'direction': 'X→Y' if consistency_metrics['regime_high_majority'] == 1 
                                           else ('Y→X' if consistency_metrics['regime_high_majority'] == -1 else 'Uncertain'),
                                'confidence': f"{consistency_metrics['regime_high_consistency']:.0f}%"
                            }
                        ]
                    }
                    st.session_state.llm_conclusion = majority_conclusion
                else:
                    # Fallback to first valid conclusion
                    valid_conclusions = [c for c in all_trial_conclusions if c is not None]
                    st.session_state.llm_conclusion = valid_conclusions[0] if valid_conclusions else None
            else:
                # Single trial: use its conclusion directly
                st.session_state.llm_conclusion = all_trial_conclusions[0] if all_trial_conclusions else None

    # Display results
    if 'llm_results' in st.session_state and st.session_state.llm_results:
        # Show which configuration was used
        if 'llm_config' in st.session_state:
            st.info(f"🎯 **Scenario**: {st.session_state.llm_config.get('scenario_name', 'Custom')}")
        
        # Check if single or multiple trials
        n_trials_run = len(st.session_state.get('llm_all_trials', []))
        
        if n_trials_run == 1:
            st.subheader("🔍 LLM Reasoning Steps")
        else:
            st.subheader(f"🔍 LLM Reasoning Steps (Showing Trial 1 of {n_trials_run})")
        
        for result in st.session_state.llm_results:
            with st.expander(f"**{result['step']}**", expanded=False):
                # Show the prompt that was sent (for debugging)
                if 'prompt_sent' in result:
                    st.markdown("#### 📤 Prompt Sent to LLM:")
                    st.code(result['prompt_sent'], language="markdown")
                    st.markdown("---")
                
                # Show the response
                st.markdown("#### 📥 LLM Response:")
                st.markdown(result['response'])
        
        # Display consistency metrics ONLY if multiple trials were run
        if 'llm_consistency' in st.session_state and n_trials_run > 1:
            st.subheader("📊 Consistency Across Trials")
            consistency = st.session_state.llm_consistency
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Regime Low Consistency", f"{consistency['regime_low_consistency']:.0f}%")
            with col2:
                st.metric("Regime High Consistency", f"{consistency['regime_high_consistency']:.0f}%")
            with col3:
                st.metric("Overall Consistency", f"{consistency['overall_consistency']:.0f}%")
            
            st.info(f"💡 Majority vote from {consistency['n_trials']} trials used for final prediction")
            
            # Show all trial results in expander
            if 'llm_all_trials' in st.session_state and len(st.session_state.llm_all_trials) > 1:
                with st.expander("🔬 View All Trial Results"):
                    for i, trial_results in enumerate(st.session_state.llm_all_trials):
                        st.markdown(f"### Trial {i+1}")
                        if trial_results:
                            # Show only final conclusion for brevity
                            final_step = trial_results[-1]
                            st.markdown(f"**{final_step['step']}**")
                            st.markdown(final_step['response'][:500] + "..." if len(final_step['response']) > 500 else final_step['response'])
                        st.divider()
        
        st.success("✅ LLM analysis complete.")
