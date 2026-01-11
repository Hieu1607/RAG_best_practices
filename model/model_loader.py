import torch
import platform
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoConfig

class ModelLoader:
    """
    Responsible for loading a specific language model and its associated tokenizer.

    Attributes:
        model_name (str): The name of the loaded model.
        model_type (str): The type of the model ('causal', 'seq2seq', 'classification').
        model (transformers.PreTrainedModel): The loaded language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
    """

    def __init__(self, model_name, model_type, quant_type=None):
        """
        Initializes the ModelLoader with a specified model name, model type, and optional quantization.

        Args:
            model_name (str): The name of the model to be loaded.
            model_type (str): The type of the model to be loaded ('causal', 'seq2seq').
            quant_type (str, optional): Type of quantization ('8bit', '4bit', or None).
        """
        self.model_name = model_name
        self.model_type = model_type
        if self.model_name != 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            # Load the model based on the type
            model_loader_function = {
                'causal': AutoModelForCausalLM,
                'seq2seq': AutoModelForSeq2SeqLM
            }.get(model_type)
        
            if not model_loader_function:
                raise ValueError(f"Unsupported model type: {model_type}")
        
            if quant_type:
                print(f"⚠️  Quantization requested ({quant_type}), but disabled to avoid cuBLAS errors.")
                print(f"   Loading {model_type} model with FP16 inference instead.")
            
            # Prepare kwargs for model loading with FP16 precision
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_name,
                'torch_dtype': torch.float16,  # Use FP16 for reduced memory usage
                'device_map': 'auto',  # Auto-assign model layers to available devices (GPU/CPU)
            }
        
            self.model = model_loader_function.from_pretrained(**model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            
            # Log model loading info
            quant_status = quant_type if quant_type and bnb_config else "No quantization"
            print(f"✓ Loaded {model_type} model: {self.model_name}")
            print(f"  - Quantization: {quant_status}")
            
            # Check device placement
            if hasattr(self.model, 'hf_device_map'):
                devices = set(self.model.hf_device_map.values())
                if any('cuda' in str(d) for d in devices):
                    gpu_devices = [d for d in devices if 'cuda' in str(d)]
                    print(f"  - Device: GPU {gpu_devices}")
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"  - GPU Memory: {memory_allocated:.2f} GB allocated")
                else:
                    print(f"  - Device: {devices}")
            else:
                device_info = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'Unknown'
                print(f"  - Device: {device_info}")
            
        else:
            # Mixtral-8x7B requires special handling with offloading
            print('Loading Mixtral-8x7B (Instruct45B) with offloading...')
            
            # Try to import mixtral-offloading modules (only works on Linux + GPU)
            try:
                import sys
                sys.path.append("mixtral-offloading")
                from src.build_model import OffloadConfig, QuantConfig, build_model
                
                # Use the specialized build_model function
                model_name = self.model_name
                self.model = build_model(
                    model_name=model_name,
                    device_map="auto",
                    offload_folder="offload",
                )
                print("Successfully loaded with mixtral-offloading")
            except (ImportError, ModuleNotFoundError) as e:
                # Fallback to standard loading if mixtral-offloading not available
                print(f"Warning: mixtral-offloading not available ({e}). Using standard loading.")
                print("Note: This may require significant GPU memory.")
                model_name = self.model_name
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",  
                    offload_folder="offload",
                    torch_dtype=torch.float16,
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')