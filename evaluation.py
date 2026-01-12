import gc
import json
import os
import argparse
from datetime import datetime
import random
import time

# Set PyTorch CUDA memory configuration before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

from model.index_builder import IndexBuilder
from model.language_model import LanguageModel
from model.model_loader import ModelLoader
from model.rag import RAG
from model.retriever import Retriever

from config import configs_run1, configs_run2, configs_test_suite

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RAG model")
    parser.add_argument('--dataset', default='truthfulqa', type=str, help='Dataset to evaluate on') 
    parser.add_argument('--output-dir', default='outputs', type=str, help='Output directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--quant', default=None, type=str, choices=['4bit', '8bit', None], help='Quantization type (4bit, 8bit, or None)')
    parser.add_argument('--num-samples', default=None, type=int, help='Number of test samples (for quick demo/testing)')
    parser.add_argument('--config-set', default='test_suite', type=str, choices=['test_suite', 'run1', 'run2'], 
                        help='Which config set to run: test_suite (all combinations), run1, or run2')
    return parser.parse_args()

# Set random seed
def set_random_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch for CPU
    torch.cuda.manual_seed(seed)  # PyTorch for GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    
    # This ensures that your code will be as deterministic as possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Initialize index builder
def initialize_index_builder(knowledge_base, config, icl_data=None):
    """Initialize index builder with optional hybrid mode support.
    
    Args:
        knowledge_base: Main knowledge base (articles)
        config: Configuration dict
        icl_data: ICL examples DataFrame (for hybrid mode)
    """
    hybrid_kb = config['index_builder'].get('hybrid_kb', False)
    
    if hybrid_kb:
        # Hybrid mode: pass both knowledge bases
        index_builder = IndexBuilder(
            knowledge_base, 
            config['embedding_model_name'], 
            config['ralm']['expand_query'], 
            **{k: v for k, v in config['index_builder'].items() if k not in ['hybrid_kb']},
            hybrid_kb=True,
            icl_df=icl_data
        )
        return index_builder.initialize_components()  # Returns 5 items
    else:
        # Normal mode
        index_builder = IndexBuilder(
            knowledge_base, 
            config['embedding_model_name'], 
            config['ralm']['expand_query'], 
            **config['index_builder']
        )
        return index_builder.initialize_components()  # Returns 3 items

# Initialize RAG model
def initialize_rag(knowledge_base, config, model_loader_generation, model_loader_seq2seq, index_pre, same_index, first_run, icl_data=None):
    build_index = not same_index or first_run
    hybrid_kb = config['index_builder'].get('hybrid_kb', False)

    # Initialize index builder if needed
    if build_index:
        if hybrid_kb:
            # Hybrid mode: returns 5 items
            index, index_icl, index_titles, doc_info, icl_info = initialize_index_builder(knowledge_base, config, icl_data)
        else:
            # Normal mode: returns 3 items
            index, index_titles, doc_info = initialize_index_builder(knowledge_base, config)
            index_icl, icl_info = None, None
    else:
        if hybrid_kb:
            index, index_icl, index_titles, doc_info, icl_info = index_pre
        else:
            index, index_titles, doc_info = index_pre
            index_icl, icl_info = None, None
    
    # Initialize retriever with hybrid mode support
    retriever = Retriever(
        index, doc_info, config['embedding_model_name'], 
        model_loader_seq2seq, index_titles, 
        index_icl=index_icl, icl_info=icl_info
    )
    
    language_model = LanguageModel(model_loader_generation, config['is_chat_model'], config['instruct_tokens'])
    
    if not same_index:
        del index, index_titles, doc_info
        if hybrid_kb:
            del index_icl, icl_info
        gc.collect()
        index_pre = None
    else:
        if hybrid_kb:
            index_pre = (index, index_icl, index_titles, doc_info, icl_info)
        else:
            index_pre = (index, index_titles, doc_info)
    
    return RAG(retriever, language_model, **config['ralm']), index_pre

    
def mean_metrics_item(evaluation):
    metrics = ['r1f1','r2f1','rLf1', 'similarity']
        
    # Initialize the dictionary to store computed means
    computed_means = {}

    # Compute means and populate the dictionary
    for metric in metrics:
        computed_means[metric] = float(evaluation[metric].mean())

    return computed_means
    
if __name__ == "__main__":

    args = parse_args()
    set_random_seed(args.seed)

    if args.dataset == 'truthfulqa':
        truthful_qa = load_dataset("truthful_qa", "generation", split='validation').to_pandas()
        test_data = truthful_qa[['question', 'best_answer', 'correct_answers', 'incorrect_answers']].copy()
        test_data['correct_answers'] = test_data['correct_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
        test_data['correct_answers'] = test_data['correct_answers'].apply(lambda x: [i for i in x if i]) # Remove empty strings from correct answers
        test_data['incorrect_answers'] = test_data['incorrect_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
        test_data['incorrect_answers'] = test_data['incorrect_answers'].apply(lambda x: [i for i in x if i])
        test_data['best_answer'] = test_data['best_answer'].apply(lambda x: [x] if x else [])
        test_data = test_data[(test_data['correct_answers'].apply(len) > 1) & (test_data['incorrect_answers'].apply(len) > 1)]
        test_data = test_data.reset_index(drop=True)
        
        # Limit samples if specified (for quick testing/demo)
        if args.num_samples:
            test_data = test_data.head(args.num_samples)
            print(f"Demo mode: Using only {args.num_samples} questions from TruthfulQA")

    elif args.dataset == 'mmlu':
        mmlu = load_dataset("cais/mmlu", "all")
        test_data = mmlu['test'].to_pandas().groupby('subject').head(32).drop(columns='subject').reset_index(drop=True)
        test_data['aswer'] = test_data['answer'].astype(int)
        test_data['choices'] = test_data['choices'].apply(lambda x: x.tolist())
        def extract_answers(row):
            correct_answers = [choice for i, choice in enumerate(row['choices']) if i == row['answer']]
            assert len(correct_answers) > 0
            incorrect_answers = [choice for i, choice in enumerate(row['choices']) if i != row['answer']]
            return pd.Series([correct_answers, incorrect_answers], index=['correct_answers', 'incorrect_answers'])
        test_data[['correct_answers', 'incorrect_answers']] = test_data.apply(extract_answers, axis=1)
        test_data = test_data.drop(columns=['choices', 'answer'])
        test_data['best_answer'] = [[] for _ in range(len(test_data))]
        test_data = test_data[['question', 'best_answer', 'correct_answers', 'incorrect_answers']]
        test_data = test_data.reset_index(drop=True)
        
        # Limit samples if specified (for quick testing/demo)
        if args.num_samples:
            test_data = test_data.head(args.num_samples)
            print(f"Demo mode: Using only {args.num_samples} questions from MMLU")
        else:
            print(f"Loaded {len(test_data)} questions from MMLU dataset")

    
    knowledge_base = pd.read_pickle('resources/articles_l3.pkl')
    all_results = {}

    # Select config set based on argument
    if args.config_set == 'test_suite':
        config_runs = [(configs_test_suite, 'test_suite')]
    elif args.config_set == 'run1':
        config_runs = [(configs_run1, 1)]
    elif args.config_set == 'run2':
        config_runs = [(configs_run2, 2)]
    else:
        config_runs = [(configs_run1, 1), (configs_run2, 2)]

    # Evaluate all configurations
    for configs, run in config_runs:
        run_timestamp = datetime.now().strftime("%m-%d_%H-%M")
        results_dir = f'{args.output_dir}/{args.dataset}/run{run}_{run_timestamp}'

        os.makedirs(results_dir, exist_ok=True)
        index_configs = [c['index_builder'] for c in configs.values()]
        same_index = all(ic == index_configs[0] for ic in index_configs)
        index_pre = None
        first_run = True
        
        evaluations = {}
        timing_results = {}
        for name, config in configs.items():
            config_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Configuration: {name}")
            print(f"Quantization: {args.quant if args.quant else 'None'}")
            print(f"GPU Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"{'='*60}\n")
            
            # Initialize model loaders with optional quantization
            load_start = time.time()
            model_loader_generation = ModelLoader(config['generation_model_name'], 'causal', quant_type=args.quant)
            model_loader_seq2seq = ModelLoader(config['seq2seq_model_name'], 'seq2seq', quant_type=args.quant)
            load_time = time.time() - load_start
            print(f"Model loading time: {load_time:.2f}s")
            
            # Load knowledge base
            if config['ralm']['icl_kb']:
                kb = test_data
            elif config['ralm']['kb_10K']:
                kb = pd.read_pickle('./resources/articles_l4.pkl')
            else:
                kb = knowledge_base
            
            # For hybrid mode, pass both kb and test_data as ICL
            icl_data = test_data if config['index_builder'].get('hybrid_kb', False) else None
            
            # Initialize RAG with timing
            init_start = time.time()
            ralm, index_pre = initialize_rag(kb, config, model_loader_generation, model_loader_seq2seq, index_pre, same_index, first_run, icl_data)
            init_time = time.time() - init_start
            print(f"RAG initialization time: {init_time:.2f}s")
            
            # Evaluate with timing
            print(f"Evaluating model: {name}")
            eval_start = time.time()
            evaluations[name], mauve_score = ralm.evaluate(test_data)
            eval_time = time.time() - eval_start
            print(f"Evaluation time: {eval_time:.2f}s")
        
            del ralm
            del model_loader_generation
            del model_loader_seq2seq
            gc.collect()
            torch.cuda.empty_cache()
            first_run = False
            
            # Calculate total time for this config
            config_total_time = time.time() - config_start_time
            timing_results[name] = {
                'model_load_time': load_time,
                'rag_init_time': init_time,
                'evaluation_time': eval_time,
                'total_time': config_total_time
            }
            
            print(f"\n{'─'*60}")
            print(f"Configuration '{name}' completed in {config_total_time:.2f}s ({config_total_time/60:.2f} min)")
            print(f"  - Model loading: {load_time:.2f}s")
            print(f"  - RAG initialization: {init_time:.2f}s")
            print(f"  - Evaluation: {eval_time:.2f}s")
            print(f"{'─'*60}\n")
            
            # Save evaluation results
            evaluations[name].to_pickle(os.path.join(results_dir, f'evaluation_{name}.pkl'))
            with open(os.path.join(results_dir, f'config_{name}.json'), 'w') as f:
                json.dump(configs[name], f, indent=4)
        
            results = mean_metrics_item(evaluations[name])
            results['mauve'] = mauve_score
            results['timing'] = timing_results[name]

            with open(f"{results_dir}/eval_results_{name}.json", "w") as outfile: 
                json.dump(results, outfile, indent=4)   
            all_results[name] = results
        del index_pre
        
        # Save timing summary for this run
        with open(f"{results_dir}/timing_summary.json", "w") as outfile:
            json.dump(timing_results, outfile, indent=4)
                
        with open(f"{results_dir}/eval_results_all.json", "w") as outfile: 
            json.dump(all_results, outfile, indent=4)
        
        # Print run summary
        run_total_time = sum(t['total_time'] for t in timing_results.values())
        print(f"\n{'='*60}")
        print(f"Run {run} Summary:")
        print(f"Total time: {run_total_time:.2f}s ({run_total_time/60:.2f} min)")
        print(f"Configurations evaluated: {len(timing_results)}")
        print(f"Average time per config: {run_total_time/len(timing_results):.2f}s")
        print(f"{'='*60}\n")   
