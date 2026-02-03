"""
Data loading utilities - Connected to existing DATA/datasets.py
"""
import os
import pandas as pd
import sys
from typing import Dict, List, Any

# Add paths for existing code imports
sys.path.append('/app')
sys.path.append('/app/DATA')
sys.path.append('/app/app/utils')

from DATA.datasets import Custom
from app.utils.ollama_client import OllamaClient

class DatasetService:
    """Service to load datasets using existing Custom loader"""
    
    def __init__(self, data_path: str = '/app/DATA/custom_pairs'):
        self.data_path = data_path
        self.ollama_client = OllamaClient()  # Will automatically detect OpenAI API key  # Initialize Ollama client
    
    def check_llm_status(self) -> bool:
        """Check if LLM is available"""
        return self.ollama_client.check_connection()
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets from custom_pairs directory"""
        datasets = []
        
        if not os.path.exists(self.data_path):
            return datasets
            
        try:
            for file in os.listdir(self.data_path):
                if file.endswith('.txt') and file != 'pairs.txt':
                    dataset_id = file.replace('.txt', '')
                    datasets.append({
                        'id': dataset_id,
                        'name': dataset_id.replace('_', ' ').title(),
                        'file': file
                    })
        except Exception as e:
            print(f"Error scanning datasets directory: {e}")
            
        return datasets

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a specific dataset using existing Custom loader"""
        try:
            dataset = Custom(pair_id=dataset_id, path=self.data_path)
            return dataset.dataframe
        except Exception as e:
            raise Exception(f"Failed to load dataset {dataset_id}: {e}")
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        try:
            dataset = Custom(pair_id=dataset_id, path=self.data_path)
            df = dataset.dataframe
            
            return {
                'id': dataset_id,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'summary': df.describe().to_dict()
            }
        except Exception as e:
            return {'error': str(e)}

    def load_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Load metadata from metadata.txt file and generate description using LLM"""
        metadata = {
            'description': f'Dataset: {dataset_id}',
            'documentation': None,
            'sources': [],
            'papers': [],
            'files_found': []
        }
        
        # Look for metadata.txt file in papers-and-files directory
        doc_path = os.path.join(self.data_path, 'papers-and-files', dataset_id)
        metadata_file = os.path.join(doc_path, 'metadata.txt')
        
        if os.path.exists(metadata_file):
            try:
                metadata['files_found'].append('metadata.txt')
                
                # Read metadata.txt content
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    content = f.read()[:1000]  # First 1000 chars to keep it fast
                
                metadata['papers'].append('metadata.txt')
                
                # Generate LLM description
                metadata['documentation'] = self._generate_llm_description(dataset_id, content, [])
                
            except Exception as e:
                metadata['documentation'] = f"Error loading metadata.txt: {e}"
        else:
            # Fallback: check if directory exists and list files
            if os.path.exists(doc_path):
                try:
                    files = [f for f in os.listdir(doc_path) if not f.endswith(':Zone.Identifier')]
                    metadata['files_found'] = files
                    metadata['documentation'] = f"No metadata.txt found. Available files: {', '.join(files[:3])}"
                except Exception:
                    metadata['documentation'] = f"Dataset directory exists but no metadata.txt found"
            else:
                metadata['documentation'] = f"No documentation directory found for {dataset_id}"
        
        return metadata
    
    def _generate_llm_description(self, dataset_id: str, paper_content: str, sources: list) -> str:
        """Generate dataset description using Ollama client"""
        try:
            # Create context for better LLM understanding
            context = {
                'dataset_name': dataset_id,
                'sources': sources[:3],  # First 3 sources
                'content_preview': paper_content[:500]  # First 500 chars
            }
            
            # Create prompt for LLM
            prompt = f"""Please provide a concise 2-3 sentence description of this dataset: {dataset_id}

Based on the following information:

Sources: {', '.join(sources[:3])}

Research content: {paper_content[:1000]}

Focus on: what variables are measured, the relationship being studied, and the main findings or purpose."""

            # Use OllamaClient instead of direct requests
            response = self.ollama_client.generate_response(
                prompt=prompt,
                context=context,
                timeout=30
            )
            
            # Check if response contains error message
            if "Error communicating with Ollama" in response:
                return f'Dataset: {dataset_id.replace("_", " ").title()} (LLM unavailable)'
            
            return response
                
        except Exception as e:
            return f'Dataset: {dataset_id.replace("_", " ").title()} (LLM unavailable: {str(e)[:50]})'# Global instance for easy import
dataset_service = DatasetService()