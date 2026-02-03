"""Batch Run Panel - Run multiple datasets with multiple methods"""
import streamlit as st
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
import json
import datetime
import io
import zipfile

# Add paths for imports
sys.path.append('/app')
sys.path.append('/app/helpers')

from app.utils.data_loader import DatasetService
from app.utils.ground_truth import (
    get_ground_truth,
    get_variable_names,
    map_prediction_to_numeric,
    map_numeric_to_string
)
from app.utils.evaluation import calculate_regime_accuracy, generate_comparison_table
from helpers.causal_param_optimization import run_segmented_causal_switch
from helpers.visualization_segmentation import segment_data
import numpy as np


def run_causal_method(method_name: str, df: pd.DataFrame, segmentation_strategy: str, segmentation_kwargs: dict) -> List[Tuple]:
    """Run a single causal method and return results"""
    try:
        # Use the run_segmented_causal_switch function
        method_lower = method_name.lower()
        
        # Get actual variable names (first two numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns[:2]
        cause_col = numeric_cols[0]
        effect_col = numeric_cols[1]
        
        # Run causal analysis - function will handle segmentation internally
        results = run_segmented_causal_switch(
            df=df.copy(),
            method=method_lower,
            segmentation_strategy=segmentation_strategy,
            segmentation_kwargs=segmentation_kwargs,
            device='cpu',
            cause_col=cause_col,
            effect_col=effect_col
        )
        
        return results if results else []
        
    except Exception as e:
        st.error(f"Error running {method_name}: {str(e)}")
        return []


def run_llm_analysis(df: pd.DataFrame, dataset_name: str, config=None) -> Dict:
    """Run LLM analysis for a dataset with optional configuration.
    
    Args:
        df: Input dataframe
        dataset_name: Name of dataset
        config: ExperimentConfig instance (optional)
        
    Returns:
        Dictionary with LLM conclusions
    """
    try:
        from app.utils.llm_client import get_llm_client
        from app.utils.prompts import get_prompt_chain
        from app.utils.data_summarizer import generate_summary
        from app.utils.llm_config import ExperimentConfig
        
        # Use provided config or create default
        if config is None:
            config = ExperimentConfig()  # Default configuration
        
        # Generate summary using new modular function
        summary = generate_summary(df, config, dataset_name)
        
        # Get threshold from ground truth
        from app.utils.ground_truth import get_ground_truth
        ground_truth = get_ground_truth(dataset_name)
        threshold_var = ground_truth.get('threshold_var') if ground_truth else None
        threshold_value = ground_truth.get('threshold_val') if ground_truth else None
        
        # Run prompt chain with configuration
        llm_client = get_llm_client("openai")
        model_name = config.model_name if hasattr(config, 'model_name') else "gpt-5.2"
        temperature = config.temperature if hasattr(config, 'temperature') else 0.0
        
        # Get dataframe columns for anonymization mapping
        df_columns = df.columns[:2].tolist() if df is not None else None
        
        prompt_chain = get_prompt_chain(config, threshold_var, threshold_value, df_columns)
        
        history = ""
        results = []
        
        for step in prompt_chain:
            prompt = step["prompt"].format(summary=summary, history=history)
            response = llm_client.ask(prompt, model=model_name, temperature=temperature)
            
            if response:
                results.append({
                    "step": step["name"], 
                    "response": response,
                    "prompt_sent": prompt  # Store prompt for display
                })
                history += f"**{step['name']}**: {response}\n\n"
            else:
                return None
        
        # Parse final JSON response
        if results:
            final_response = results[-1]['response']
            
            # Strip markdown code blocks if present
            if '```json' in final_response:
                final_response = final_response.split('```json')[1].split('```')[0].strip()
            elif '```' in final_response:
                final_response = final_response.split('```')[1].split('```')[0].strip()
            
            # Log the cleaned response for debugging
            st.write(f"LLM raw response for {dataset_name}:")
            st.code(final_response, language='json')
            
            try:
                conclusion = json.loads(final_response)
                # Return both conclusion and full results (with prompts)
                return {
                    'conclusion': conclusion,
                    'full_results': results,
                    'summary': summary  # Also store the summary for reference
                }
            except json.JSONDecodeError as je:
                st.error(f"JSON parsing error: {str(je)}")
                st.error(f"Problematic JSON: {final_response}")
                return None
        
        return None
        
    except Exception as e:
        st.error(f"Error running LLM analysis for {dataset_name}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def format_step_to_txt(step_name: str, prompt_sent: str, response: str) -> str:
    """Format a single prompt step as a text file content.
    
    Args:
        step_name: Name of the step (e.g., "Step 1: Domain Understanding")
        prompt_sent: The prompt that was sent to the LLM
        response: The LLM's response
        
    Returns:
        Formatted string for the text file
    """
    content = f"{'='*80}\n"
    content += f"{step_name}\n"
    content += f"{'='*80}\n\n"
    content += f"📤 PROMPT SENT TO LLM:\n"
    content += f"{'-'*40}\n"
    content += f"{prompt_sent}\n\n"
    content += f"📥 LLM RESPONSE:\n"
    content += f"{'-'*40}\n"
    content += f"{response}\n"
    return content


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace problematic characters
    for char in [':', '/', '\\', '?', '*', '"', '<', '>', '|']:
        name = name.replace(char, '_')
    return name


def generate_prompt_logs_zip(batch_results: List[Dict]) -> Optional[bytes]:
    """Generate a ZIP file containing all LLM prompt logs.
    
    Structure depends on whether multiple configs were run:
    - Single config: {dataset_name}/Step_N_Name.txt
    - Multiple configs: {Config_Name}/{dataset_name}/Step_N_Name.txt
    
    Args:
        batch_results: List of batch result dictionaries
        
    Returns:
        ZIP file as bytes, or None if no LLM data available
    """
    # Check if we have any LLM results
    has_llm_data = False
    run_all_configs = False
    
    for result in batch_results:
        for method_name in result['methods'].keys():
            if method_name.startswith('LLM'):
                has_llm_data = True
                if method_name.startswith('LLM_'):
                    run_all_configs = True
                break
        if has_llm_data:
            break
    
    if not has_llm_data:
        return None
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in batch_results:
            dataset_name = sanitize_filename(result['dataset_name'])
            
            for method_name, method_results in result['methods'].items():
                if not method_name.startswith('LLM'):
                    continue
                
                if not isinstance(method_results, dict) or 'full_results' not in method_results:
                    continue
                
                llm_data = method_results
                full_results = llm_data.get('full_results', [])
                
                if not full_results:
                    continue
                
                # Determine folder structure
                if run_all_configs and method_name.startswith('LLM_'):
                    config_name = sanitize_filename(method_name[4:])  # Remove "LLM_" prefix
                    base_path = f"{config_name}/{dataset_name}"
                else:
                    base_path = dataset_name
                
                # Create a text file for each step
                for idx, step_result in enumerate(full_results, 1):
                    step_name = step_result.get('step', f'Step {idx}')
                    prompt_sent = step_result.get('prompt_sent', 'No prompt available')
                    response = step_result.get('response', 'No response available')
                    
                    # Create sanitized filename from step name
                    # e.g., "Step 1: Domain Understanding & Hypothesis Generation" -> "Step_1_Domain_Understanding.txt"
                    step_filename = sanitize_filename(step_name)
                    # Shorten long names
                    if len(step_filename) > 50:
                        step_filename = step_filename[:50]
                    step_filename = f"{step_filename}.txt"
                    
                    file_path = f"{base_path}/{step_filename}"
                    file_content = format_step_to_txt(step_name, prompt_sent, response)
                    
                    zip_file.writestr(file_path, file_content)
                
                # Also add the dataset summary as a separate file
                if 'summary' in llm_data:
                    summary_path = f"{base_path}/00_Dataset_Summary.txt"
                    summary_content = f"{'='*80}\n"
                    summary_content += "DATASET SUMMARY SENT TO LLM\n"
                    summary_content += f"{'='*80}\n\n"
                    summary_content += llm_data['summary']
                    zip_file.writestr(summary_path, summary_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def render():
    """Render the batch run panel"""
    st.header("🚀 Batch Run")
    st.write("Run multiple datasets with selected causal methods and generate comparison reports.")
    
    # Initialize dataset service
    dataset_service = DatasetService(data_path='/app/DATA/custom_pairs')
    available_datasets = dataset_service.get_available_datasets()
    
    if not available_datasets:
        st.warning("No datasets found in DATA/custom_pairs/")
        return
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Datasets:**")
        
        # Simple select all checkbox
        select_all = st.checkbox("Select All Datasets")
        
        # If select all is checked, update all dataset checkboxes in session state
        if select_all:
            for dataset in available_datasets:
                st.session_state[f"dataset_{dataset['id']}"] = True
        
        selected_datasets = []
        for dataset in available_datasets:
            dataset_id = dataset['id']
            dataset_name = dataset['name']
            
            # Simple checkbox - state controlled by key
            if st.checkbox(dataset_name, key=f"dataset_{dataset_id}"):
                selected_datasets.append(dataset_id)
    
    with col2:
        st.write("**Select Methods:**")
        
        run_roche_batch = st.checkbox("ROCHE (Heteroscedastic)", value=True)
        run_loci_batch = st.checkbox("LOCI (Location-Scale)", value=True)
        run_lcube_batch = st.checkbox("LCUBE (Dense Detection)", value=True)
        run_llm_batch = st.checkbox("LLM Analysis (OpenAI)", value=False)
    
    # LLM Configuration (only show if LLM is selected)
    llm_config = None
    run_all_llm_configs = False
    selected_llm_configs = []
    
    if run_llm_batch:
        with st.expander("🤖 LLM Configuration", expanded=True):
            from app.utils.llm_config import get_preset_config, get_available_presets, AVAILABLE_MODELS
            
            # Model selection
            selected_model = st.selectbox("Model:", AVAILABLE_MODELS, index=0, key="batch_model")
            
            # Run all configs checkbox
            run_all_llm_configs = st.checkbox(
                "Run All LLM Configurations",
                value=False,
                help="Run all 4 configuration presets (Raw Named, Raw Anonymous, Segmented Named, Segmented Anonymous)"
            )
            
            # Get all available presets (all are now _hint_explicit variants)
            all_presets = get_available_presets()
            
            # Create display names for the dropdown
            preset_display_names = {p: get_preset_config(p).scenario_name for p in all_presets}
            
            if run_all_llm_configs:
                # Show which configs will run
                st.info("**Will run all configurations:**\n" + "\n".join([f"- {v}" for v in preset_display_names.values()]))
                selected_llm_configs = all_presets
            else:
                # Preset selection dropdown (disabled when run_all is checked)
                selected_display_name = st.selectbox(
                    "Configuration Preset:",
                    options=list(preset_display_names.values()),
                    index=list(preset_display_names.values()).index('Raw+Stats | Named | Hint | Explicit'),  # Default
                    key="batch_preset",
                    disabled=run_all_llm_configs
                )
                
                # Get the actual preset name from display name
                preset_name = [k for k, v in preset_display_names.items() if v == selected_display_name][0]
                
                llm_config = get_preset_config(preset_name)
                
                # Update model from dropdown selection
                llm_config.model_name = selected_model
                selected_llm_configs = [preset_name]
                
                # Display configuration summary
                st.info(f"""
                **Active Configuration:** `{preset_name}`
                
                {llm_config.get_summary_description()}
                
                - 🤖 **Model**: {llm_config.model_name}
                - 🌡️ **Temperature**: {llm_config.temperature}
                """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("---")
        st.info("ℹ️ Thresholds are read from pairmeta_with_ground_truth.txt for each dataset")
    with col2:
        pass
    
    # Summary
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasets Selected", len(selected_datasets))
    with col2:
        methods_count = sum([run_roche_batch, run_loci_batch, run_lcube_batch, run_llm_batch])
        st.metric("Methods Selected", methods_count)
    with col3:
        total_runs = len(selected_datasets) * methods_count
        st.metric("Total Runs", total_runs)
    
    # Run button
    if st.button("▶ Start Batch Run", type="primary", disabled=(len(selected_datasets) == 0 or methods_count == 0)):
        run_batch_analysis(
            selected_datasets=selected_datasets,
            dataset_service=dataset_service,
            run_roche=run_roche_batch,
            run_loci=run_loci_batch,
            run_lcube=run_lcube_batch,
            run_llm=run_llm_batch,
            llm_config=llm_config,
            run_all_llm_configs=run_all_llm_configs,
            selected_llm_configs=selected_llm_configs,
            selected_model=selected_model if run_llm_batch else None
        )
    
    # Display previous batch results if available
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        display_batch_results(st.session_state.batch_results)


def run_batch_analysis(selected_datasets, dataset_service, run_roche, run_loci, run_lcube, run_llm, 
                       llm_config=None, run_all_llm_configs=False, selected_llm_configs=None, selected_model=None):
    """Run batch analysis on selected datasets with optional LLM configuration.
    
    Args:
        selected_datasets: List of dataset IDs to process
        dataset_service: DatasetService instance
        run_roche: Whether to run ROCHE algorithm
        run_loci: Whether to run LOCI algorithm
        run_lcube: Whether to run LCUBE algorithm
        run_llm: Whether to run LLM analysis
        llm_config: Single LLM config (used when run_all_llm_configs=False)
        run_all_llm_configs: Whether to run all LLM configurations
        selected_llm_configs: List of preset names to run
        selected_model: Model name to use for LLM
    """
    from app.utils.llm_config import get_preset_config
    
    st.write("---")
    st.subheader("📊 Running Batch Analysis...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_datasets = len(selected_datasets)
    batch_results = []
    
    for idx, dataset_id in enumerate(selected_datasets):
        status_text.text(f"Processing: {dataset_id} ({idx + 1}/{total_datasets})")
        progress_bar.progress((idx) / total_datasets)
        
        try:
            # Load dataset
            df = dataset_service.load_dataset(dataset_id)
            
            # Get variable names first
            cause_var, effect_var = get_variable_names(dataset_id, df)
            
            # Get ground truth to retrieve threshold information
            ground_truth = get_ground_truth(dataset_id)
            
            if not ground_truth:
                st.warning(f"⚠️ No ground truth found for {dataset_id} in pairmeta file. Skipping...")
                continue
            
            # Ensure only 2 columns
            numeric_cols = df.select_dtypes(include=['number']).columns[:2]
            df_numeric = df[numeric_cols].copy()
            
            # Get threshold from ground truth
            threshold_var = ground_truth['threshold_var']
            threshold_value = ground_truth['threshold_val']
            
            # Find the column that matches the threshold variable
            threshold_col = None
            for col in numeric_cols:
                if threshold_var.lower() in col.lower() or col.lower() in threshold_var.lower():
                    threshold_col = col
                    break
            
            if not threshold_col:
                st.warning(f"⚠️ Could not find column matching threshold variable '{threshold_var}' for {dataset_id}. Using first numeric column.")
                threshold_col = numeric_cols[0]
            
            st.write(f"📍 {dataset_id}: Using threshold {threshold_col} @ {threshold_value}")
            
            # Prepare segmentation parameters
            segmentation_strategy = 'threshold'
            segmentation_kwargs = {
                'column': threshold_col,
                'threshold': threshold_value,
                'plot_results': False
            }
            
            # Initialize result for this dataset
            dataset_result = {
                'dataset_name': dataset_id,
                'cause_var': cause_var,
                'effect_var': effect_var,
                'threshold_col': threshold_col,
                'threshold_value': threshold_value,
                'methods': {},
                'ground_truth': ground_truth,
                'evaluation': {},
                'llm_configs_used': [],  # Track which LLM configs were used
                'run_all_llm_configs': run_all_llm_configs  # Flag for export formatting
            }
            
            # Run causal methods
            if run_roche:
                roche_results = run_causal_method('ROCHE', df_numeric, segmentation_strategy, segmentation_kwargs)
                if roche_results:
                    # Sort by segment ID: 0=low, 1=high
                    roche_results = sorted(roche_results, key=lambda x: x[0])
                    dataset_result['methods']['ROCHE'] = roche_results
                    st.write(f"✓ ROCHE - Segment 0 (Low): {roche_results[0][1]}, Segment 1 (High): {roche_results[1][1] if len(roche_results) > 1 else 'N/A'}")
            
            if run_loci:
                loci_results = run_causal_method('LOCI', df_numeric, segmentation_strategy, segmentation_kwargs)
                if loci_results:
                    loci_results = sorted(loci_results, key=lambda x: x[0])
                    dataset_result['methods']['LOCI'] = loci_results
                    st.write(f"✓ LOCI - Segment 0 (Low): {loci_results[0][1]}, Segment 1 (High): {loci_results[1][1] if len(loci_results) > 1 else 'N/A'}")
            
            if run_lcube:
                lcube_results = run_causal_method('LCUBE', df_numeric, segmentation_strategy, segmentation_kwargs)
                if lcube_results:
                    lcube_results = sorted(lcube_results, key=lambda x: x[0])
                    dataset_result['methods']['LCUBE'] = lcube_results
                    st.write(f"✓ LCUBE - Segment 0 (Low): {lcube_results[0][1]}, Segment 1 (High): {lcube_results[1][1] if len(lcube_results) > 1 else 'N/A'}")
            
            # Run LLM analysis
            if run_llm:
                # Determine which configs to run
                configs_to_run = selected_llm_configs if selected_llm_configs else []
                
                for preset_name in configs_to_run:
                    config = get_preset_config(preset_name)
                    if selected_model:
                        config.model_name = selected_model
                    
                    # Create a unique method key for this config
                    if run_all_llm_configs:
                        method_key = f"LLM_{preset_name}"
                    else:
                        method_key = "LLM"
                    
                    st.write(f"🤖 Running LLM with config: {config.scenario_name}")
                    
                    llm_results = run_llm_analysis(df_numeric, dataset_id, config)
                    if llm_results:
                        # Store full results (conclusion + prompts/outputs) with config info
                        llm_results['config_name'] = preset_name
                        llm_results['config'] = config.to_dict()
                        dataset_result['methods'][method_key] = llm_results
                        dataset_result['llm_configs_used'].append(preset_name)
                        # For display, show the conclusion
                        conclusion = llm_results.get('conclusion', {})
                        st.write(f"✓ {method_key} results for {dataset_id}: {conclusion}")
                    else:
                        st.warning(f"{method_key} analysis returned no results for {dataset_id}")
            
            # Evaluate against ground truth if available
            if ground_truth:
                all_predictions = {}
                
                for method_name, results in dataset_result['methods'].items():
                    # Check if this is a non-LLM method (list of tuples)
                    if not method_name.startswith('LLM') and isinstance(results, list) and len(results) >= 2:
                        # ✅ DIRECT MAPPING: Segment 0 = regime_low, Segment 1 = regime_high
                        segment_0_dir = results[0][1]  # (segment_id, direction, score)
                        segment_1_dir = results[1][1] if len(results) > 1 else ''
                        
                        all_predictions[method_name] = {
                            'regime_low': map_prediction_to_numeric(segment_0_dir, cause_var, effect_var),
                            'regime_high': map_prediction_to_numeric(segment_1_dir, cause_var, effect_var)
                        }
                        
                        st.write(f"📊 {method_name} - Low: {segment_0_dir}, High: {segment_1_dir}")
                        
                    elif method_name.startswith('LLM'):
                        # LLM returns dict with conclusion and full_results
                        llm_data = results if isinstance(results, dict) else {'conclusion': results}
                        conclusion = llm_data.get('conclusion', results)
                        config_used = llm_data.get('config', {})
                        directions = conclusion.get('directions', [])
                        if len(directions) >= 2:
                            # ✅ FIXED: Map LLM regime numbers (0=low, 1=high) to regime_low and regime_high
                            regime_directions = {d.get('regime', d.get('cluster', idx)): d.get('direction', '') for idx, d in enumerate(directions)}
                            
                            # Get directions for regime 0 (low) and regime 1 (high)
                            dir_low = regime_directions.get(0, '')
                            dir_high = regime_directions.get(1, '')
                            
                            # Fallback: if regime numbers don't match, use order
                            if not dir_low and len(directions) > 0:
                                dir_low = directions[0].get('direction', '')
                            if not dir_high and len(directions) > 1:
                                dir_high = directions[1].get('direction', '')
                            
                            # Handle anonymized variable mapping based on config used for this run
                            is_anonymized = config_used.get('raw_data_options', {}).get('anonymize_names', False)
                            if is_anonymized:
                                # For anonymized data, LLM uses Variable_X and Variable_Y
                                # Map them back to actual variable names for evaluation
                                all_predictions[method_name] = {
                                    'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var, use_anonymized=True),
                                    'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var, use_anonymized=True)
                                }
                                st.write(f"📊 {method_name} (anonymized): Low=C0 ({dir_low}), High=C1 ({dir_high})")
                            else:
                                # Non-anonymized: use original mapping
                                all_predictions[method_name] = {
                                    'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var),
                                    'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var)
                                }
                                st.write(f"📊 {method_name}: Low=C0 ({dir_low}), High=C1 ({dir_high})")
                
                # Calculate accuracy for each method
                for method_name, predictions in all_predictions.items():
                    accuracy_result = calculate_regime_accuracy(predictions, ground_truth)
                    dataset_result['evaluation'][method_name] = accuracy_result['overall_accuracy']
                    st.write(f"📈 {method_name} accuracy: {accuracy_result['overall_accuracy']*100:.0f}%")
            
            batch_results.append(dataset_result)
            
        except Exception as e:
            st.error(f"Error processing {dataset_id}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"✓ Completed {total_datasets} datasets")
    
    # Store results in session state
    st.session_state.batch_results = batch_results
    
    st.success(f"✓ Batch analysis complete! Processed {len(batch_results)} datasets.")


def display_batch_results(batch_results):
    """Display batch results in a comprehensive table"""
    
    st.write("---")
    st.subheader("📈 Batch Results")
    
    # Create summary table
    summary_rows = []
    
    for result in batch_results:
        dataset_name = result['dataset_name']
        methods = result['methods']
        ground_truth = result['ground_truth']
        evaluation = result['evaluation']
        
        row = {
            'Dataset': dataset_name,
            'Has Ground Truth': '✓' if ground_truth else '✗',
            'Threshold': f"{result.get('threshold_col', 'N/A')} @ {result.get('threshold_value', 0):.1f}"
        }
        
        # Add method predictions (Segment 0 = Low, Segment 1 = High)
        for method_name, method_results in methods.items():
            # Check for non-LLM methods (list of tuples)
            if not method_name.startswith('LLM') and isinstance(method_results, list):
                # Sort by segment ID
                sorted_results = sorted(method_results, key=lambda x: x[0])
                if len(sorted_results) >= 1:
                    row[f'{method_name} Low'] = sorted_results[0][1]  # Segment 0
                if len(sorted_results) >= 2:
                    row[f'{method_name} High'] = sorted_results[1][1]  # Segment 1
            elif method_name.startswith('LLM'):
                # LLM returns dict with conclusion and full_results
                llm_data = method_results if isinstance(method_results, dict) else {'conclusion': method_results}
                conclusion = llm_data.get('conclusion', method_results)
                directions = conclusion.get('directions', [])
                
                # Use short display name for column headers
                display_name = method_name  # e.g., "LLM" or "LLM_raw_named_hint_explicit"
                if method_name.startswith('LLM_'):
                    # Shorten config name for display
                    config_suffix = method_name[4:]  # Remove "LLM_" prefix
                    display_name = f"LLM ({config_suffix.replace('_hint_explicit', '')})"
                
                if len(directions) >= 2:
                    for idx, dir_info in enumerate(directions):
                        regime_num = dir_info.get('regime', dir_info.get('cluster', idx))
                        direction = dir_info.get('direction', 'N/A')
                        confidence = dir_info.get('confidence', 'N/A')
                        if regime_num == 0:
                            regime_name = 'Low'
                        elif regime_num == 1:
                            regime_name = 'High'
                        else:
                            regime_name = f"R{regime_num}"
                        row[f'{display_name} {regime_name}'] = f"{direction} ({confidence})"
                else:
                    row[f'{display_name}'] = 'No directions found'
        
        # Add accuracy if ground truth available
        if evaluation:
            for method_name, accuracy in evaluation.items():
                # Shorten LLM config names in accuracy columns
                if method_name.startswith('LLM_'):
                    config_suffix = method_name[4:]
                    display_name = f"LLM ({config_suffix.replace('_hint_explicit', '')})"
                else:
                    display_name = method_name
                row[f'{display_name} Acc'] = f"{accuracy * 100:.0f}%"
        
        summary_rows.append(row)
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)
        
        # Display LLM prompts and outputs if available
        # Find all LLM method keys (could be 'LLM' or 'LLM_config_name')
        llm_method_keys = set()
        for result in batch_results:
            for method_name in result['methods'].keys():
                if method_name.startswith('LLM'):
                    llm_method_keys.add(method_name)
        
        if llm_method_keys:
            st.write("---")
            st.subheader("🤖 LLM Prompts & Outputs")
            
            for result in batch_results:
                dataset_name = result['dataset_name']
                
                for method_key in sorted(llm_method_keys):
                    if method_key not in result['methods']:
                        continue
                    
                    llm_data = result['methods'][method_key]
                    
                    # Check if we have full_results with prompts
                    if isinstance(llm_data, dict) and 'full_results' in llm_data:
                        config_name = llm_data.get('config_name', 'default')
                        display_label = f"📄 {dataset_name} - {config_name}" if method_key != 'LLM' else f"📄 {dataset_name}"
                        
                        with st.expander(display_label, expanded=False):
                            # Show config used
                            if llm_data.get('config'):
                                st.info(f"🎯 **Configuration**: {llm_data['config'].get('scenario_name', config_name)}")
                            
                            # Show dataset summary
                            if 'summary' in llm_data:
                                with st.expander("📊 Dataset Summary Sent to LLM", expanded=False):
                                    st.code(llm_data['summary'], language="markdown")
                            
                            # Show each reasoning step
                            for step_result in llm_data['full_results']:
                                with st.expander(f"**{step_result['step']}**", expanded=False):
                                    # Show the prompt that was sent
                                    if 'prompt_sent' in step_result:
                                        st.markdown("#### 📤 Prompt Sent to LLM:")
                                        st.code(step_result['prompt_sent'], language="markdown")
                                        st.markdown("---")
                                    
                                    # Show the response
                                    st.markdown("#### 📥 LLM Response:")
                                    st.markdown(step_result['response'])
        
        # Calculate overall statistics
        if any(result['ground_truth'] for result in batch_results):
            st.subheader("📊 Overall Statistics")
            
            all_methods = set()
            for result in batch_results:
                all_methods.update(result['evaluation'].keys())
            
            stats_cols = st.columns(len(all_methods) if all_methods else 1)
            
            for idx, method in enumerate(sorted(all_methods)):
                accuracies = [
                    result['evaluation'].get(method, 0) 
                    for result in batch_results 
                    if result['ground_truth'] and method in result['evaluation']
                ]
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies) * 100
                    with stats_cols[idx]:
                        st.metric(
                                f"{method} Avg",
                                f"{avg_accuracy:.1f}%",
                                help=f"Average accuracy across {len(accuracies)} datasets"
                            )
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col1:
            # JSON export
            json_str = json.dumps(batch_results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"batch_results_{timestamp}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"batch_results_{timestamp}.csv",
                mime="text/csv"
            )
        
        with col3:
            # LLM Prompt Logs export (ZIP)
            prompt_zip = generate_prompt_logs_zip(batch_results)
            if prompt_zip:
                st.download_button(
                    label="📥 Download Prompt Logs (ZIP)",
                    data=prompt_zip,
                    file_name=f"llm_prompt_logs_{timestamp}.zip",
                    mime="application/zip",
                    help="Download all LLM prompts and responses as text files"
                )
            else:
                st.info("No LLM prompt logs available")
