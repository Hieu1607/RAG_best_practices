from copy import deepcopy

base_config = {
    "generation_model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "seq2seq_model_name": "google/flan-t5-small",
    "is_chat_model": True,
    "instruct_tokens": ("[INST]","[/INST]"),
    "index_builder": {
        "tokenizer_model_name": None,
        "chunk_size": 64,
        "overlap": 8,
        "passes": 10,
        "icl_kb": False,
        "multi_lingo": False
        },
    "ralm": {
        "expand_query": False,
        "top_k_docs": 1,
        "top_k_titles": 3,
        "system_prompt": "You are a truthful expert question-answering bot and should correctly and concisely answer the following question",
        "repeat_system_prompt": True,
        "stride": -1,
        "query_len": 200,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.1,
        "num_beams": 1,
        "max_new_tokens": 25,
        "batch_size": 1,
        "kb_10K": False,
        "icl_kb": False,
        "icl_kb_incorrect": False,
        "focus": False,
        "hybrid_kb": False,
        "top_k_icl": 0
        }
    }
base_config["index_builder"]["tokenizer_model_name"] = base_config["generation_model_name"]

# Generate configurations for different runs
def generate_configurations(base_config, configs):
    complete_configs = {}
    # Copy the base config and update with the specific values
    for key, config_values in configs.items():
        config = deepcopy(base_config)
        for config_key, value in config_values.items():
            if isinstance(value, dict):
                config[config_key].update(value)
            else:
                config[config_key] = value
        complete_configs[key] = config
    return complete_configs


# Comprehensive test suite for all feature combinations
configs_test_suite = {
    "1_Baseline": {
        # No special features - pure RAG baseline
    },
    "2_ExpandQuery_Only": {
        "ralm": {
            "expand_query": True,
            "top_k_docs": 3,
            "top_k_titles": 5
        }
    },
    "3_Focus_Only": {
        "ralm": {
            "top_k_docs": 10,
            "focus": 3
        }
    },
    "4_ICL_Only": {
        "index_builder": {
            "chunk_size": 200,
            "overlap": 0,
            "icl_kb": True
        },
        "ralm": {
            "top_k_docs": 2,
            "icl_kb": True,
            "icl_kb_incorrect": False
        }
    },
    "5_ExpandQuery_Focus": {
        "ralm": {
            "expand_query": True,
            "top_k_docs": 10,
            "top_k_titles": 5,
            "focus": 3
        }
    },
    "6_Focus_ICL": {
        "index_builder": {
            "chunk_size": 200,
            "overlap": 0,
            "icl_kb": True
        },
        "ralm": {
            "top_k_docs": 10,
            "focus": 3,
            "icl_kb": True,
            "icl_kb_incorrect": False
        }
    },
    "7_Hybrid_All_Features": {
        "index_builder": {
            "hybrid_kb": True,
            "chunk_size": 64,
            "overlap": 8
        },
        "ralm": {
            "hybrid_kb": True,
            "top_k_icl": 2,
            "top_k_docs": 10,
            "focus": 3,
            "expand_query": True,
            "top_k_titles": 5,
            "icl_kb_incorrect": False
        }
    }
}

configs_run1 = {
    "Base": {
    },
    "HelpV2": {
    "ralm": {
        "system_prompt": "You are an accurate and reliable question-answering bot. Please provide a precise and correct response to the question following"
        }
    },
    "Instruct45B": {
    "generation_model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "seq2seq_model_name": "google/flan-t5-small",
    "ralm": {
        "top_k_docs": 2,
        "batch_size": 4,
        "repeat_system_prompt": True,
        }
    }
}


configs_run2 = {
    "ICL1D+": {
    "index_builder": {
        "chunk_size":200,
        "overlap":0,
        "icl_kb": True
    },
    "ralm": {
        "top_k_docs": 1,
        "icl_kb": True,
        "icl_kb_incorrect": True
        }
    },
    "Focus80_Doc80": {
    "ralm": {
        "top_k_docs": 80,
        "repeat_system_prompt": True,
        "focus": 80
        }
    },
    "Hybrid_ICL2_Doc3_Focus": {
    "index_builder": {
        "hybrid_kb": True,
        "chunk_size": 64,
        "overlap": 8
    },
    "ralm": {
        "hybrid_kb": True,
        "top_k_icl": 2,
        "top_k_docs": 10,
        "focus": 3,
        "expand_query": True,
        "top_k_titles": 5,
        "icl_kb_incorrect": True
        }
    }
}



configs_test_suite = generate_configurations(base_config, configs_test_suite)



configs_run1 = generate_configurations(base_config, configs_run1)
configs_run2 = generate_configurations(base_config, configs_run2)