class ModelConfig:
    vocab_size  = 50262
    n_embd      = 512
    n_head      = 8
    n_layer     = 8
    block_size  = 128
    dropout     = 0.1


class TrainConfig:
    data_path      = "data/training_data.json"
    tokenizer_path = "tokenizer/tokenizer.json"
    checkpoint_dir = "checkpoints"
    batch_size     = 2
    grad_accum     = 2
    max_iters      = 50000
    eval_interval  = 500
    save_interval  = 200
    eval_iters     = 50
    learning_rate  = 3e-4
    weight_decay   = 0.1
    beta1          = 0.9
    beta2          = 0.95
    grad_clip      = 1.0
    warmup_iters   = 1000
    device         = "auto"
    compile_model  = False
    log_interval   = 50


class ChatConfig:
    tokenizer_path  = "tokenizer/tokenizer.json"
    max_new_tokens  = 80
    temperature     = 0.8
    top_k           = 40
    top_p           = 0.9
    device          = "auto"

    def get_checkpoint_path(self):
        import os
        for name in ["best_model.pt", "final_model.pt", "latest_model.pt"]:
            path = os.path.join("checkpoints", name)
            if os.path.exists(path):
                return path
        return None