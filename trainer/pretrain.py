import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch
import deepspeed
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model.config import MiniMindConfig
from model.modeling import MiniMindForCausalLM
from utils.preprocess import prepare_pretrain_dataset
import json

warnings.filterwarnings('ignore')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Pretraining with DeepSpeed")
    
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_swanlab", action="store_true", help="使用 SwanLab 进行日志记录")
    parser.add_argument("--swanlab_project", type=str, default="MiniMind-Pretrain-DeepSpeed")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../datasets/pretrain_hq.jsonl")
    parser.add_argument("--config_path", type=str, default="../configs/model_config.json")
    parser.add_argument("--dataset_cache_path", type=str, default="../datasets")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # --- 1. 初始化模型和分词器 ---
    with open(args.config_path, "r") as f:
        model_config = json.load(f)
        lm_config = MiniMindConfig(**model_config)
    model = MiniMindForCausalLM(lm_config)
    tokenizer = AutoTokenizer.from_pretrained('../model')

    # --- 2. 准备数据集和数据整理器 ---
    train_ds = prepare_pretrain_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        cache_dir=args.dataset_cache_path,
        num_proc=16,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # --- 3. 初始化 SwanLab 回调 ---
    swanlab_callback = SwanLabCallback(
        project="Qwen3-0.6B-from-scratch",
        experiment_name="Qwen3-0.6B-pretrain",
        description="Pretraining Qwen3-0.6B model",
        config={
            "model_name": "Qwen3-0.6B",
            "dataset": "pretrain_hq.jsonl",
            "train_size": len(train_ds),
        },
    )

    # --- 4. 设置训练参数 (替代手动的参数和deepspeed.json部分内容) ---
    training_args = TrainingArguments(
        output_dir="../output/qwen3_0.6B_pretrain_checkpoint",
        num_train_epochs=4,  # 根据需要调整
        per_device_train_batch_size=1,  # 根据你的显存调整
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        logging_steps=100,
        save_steps=1000,
        deepspeed="../configs/ds_z2_no_offload.json", # 直接传入配置文件路径
        fp16=True, # 或者 bf16=True
        report_to="none",
        optim="adamw_bnb_8bit",
    )

    # --- 5. 创建并运行 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )

    trainer.train()

    # --- 6. 保存最终模型 ---
    trainer.save_model("../output/qwen3_0.6B_final")