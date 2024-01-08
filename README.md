# LoRA_jittor
使用方法：
* 得到pytorch模型并进行测试的方法：克隆LoRA官方仓库：https://github.com/microsoft/LoRA，按照LoRA/examples/NLG/README.md的说明完成实验配置，如下载pretrained models, 数据集等，训练得到pytorch模型，并运行gpt2_beam.py用beam search算法得到模型生成的句子，用E2E数据集的evaluation方法对结果进行评估。
* 得到jittor模型并进行测试的方法：在原有基础上，将src_jittor这个目录copy到LoRA/examples/NLG目录下，之后：
1. 运行gpt2_ft_jittor.py，按需配置各参数，让gpt2模型在E2E数据集上进行finetune
例如在NLG目录下运行finetune程序可以采用如下参数（本次实验中采用了这样的参数）：
python src_jittor/gpt2_ft_jittor.py     --train_data ./data/e2e/train.jsonl     --valid_data ./data/e2e/valid.jsonl     --train_batch_size 3     --grad_acc 1     --valid_batch_size 3     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.01     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 5000     --lora_dim 4     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_jittor/GPT2_MD/e2e     --random_seed 110    --eval_interval 2000

2. 运行gpt2_beam_jittor.py，按需配置各参数，让经过finetune的gpt2模型根据测试集的输入生成句子（此时每个词还是用在字典中的位置序号表示，没有经过decode）
例如可以采用这样的参数（本次实验中采用了这样的参数）：
python src_jittor/gpt2_beam_jittor.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint /root/HW5/LoRA/examples/NLG/trained_models/GPT2_jittor/GPT2_MD/e2e/model.46000.pkl \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_jittor/GPT2_MD/e2e \
    --output_file predict.jittor.jsonl

3. 将每个词汇进行decode得到模型在测试集上的最终句子输出
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file /root/HW5/LoRA/examples/NLG/trained_models/GPT2_jittor/GPT2_MD/e2e/predict.jittor.jsonl \
    --input_file /root/HW5/LoRA/examples/NLG/data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref_jittor_MD.txt \
    --output_pred_file e2e_pred_jittor_MD.txt

4. 利用E2E数据集给出的评测程序进行评估，需要配置java环境
python eval/e2e/measure_scores.py e2e_ref_jittor_MD.txt e2e_pred_jittor_MD.txt -p
