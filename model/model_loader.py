import torch
import platform
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
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
            # Generate quantization configuration
            # Disable quantization on Windows due to bitsandbytes compatibility issues
            bnb_config = None
            is_windows = platform.system() == 'Windows'
            has_cuda = torch.cuda.is_available()
            
            if quant_type and not is_windows and has_cuda:
                try:
                    if quant_type == '8bit':
                        bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
                    elif quant_type == '4bit':
                        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                except Exception as e:
                    print(f"Warning: Quantization failed ({e}). Loading model without quantization.")
                    bnb_config = None
            elif is_windows:
                print(f"Info: Running on Windows. Quantization disabled. Loading {model_type} model without quantization.")
        
            # Load the model based on the type
            model_loader_function = {
                'causal': AutoModelForCausalLM,
                'seq2seq': AutoModelForSeq2SeqLM
            }.get(model_type)
        
            if not model_loader_function:
                raise ValueError(f"Unsupported model type: {model_type}")
        
            # Prepare kwargs for model loading
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_name,
                'quantization_config': bnb_config,
            }
        
            if model_type == "seq2seq":
                model_kwargs['device_map'] = 'auto'
        
            self.model = model_loader_function.from_pretrained(**model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            
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