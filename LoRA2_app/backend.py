from flask import Flask, render_template, request, jsonify
from model import GPT2Config, GPT2LMModel

import argparse
import time
import math
import os, sys
import json
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F


import json
import numpy as np

import encoder

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import numpy
import io
import sys
import threading
import math
import random

import json
import collections
from collections import Counter
from collections import OrderedDict


# app = Flask(__name__)

# # 模型初始化
# config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
# lm_net = GPT2LMModel(config)
# # 在这里加载预训练模型权重
# # lm_net.load_weight(pretrained_weights)
# lm_net = lm_net.cuda()
# lm_net.eval()

# # 处理用户输入并生成输出
# def generate_output(input_text):
#     input_data = torch.tensor([[lm_net.tokenizer.encode(input_text)]], dtype=torch.long).cuda()
#     with torch.no_grad():
#         output_ids = lm_net.generate(input_data)
#     output_text = lm_net.tokenizer.decode(output_ids[0])
#     return output_text

# # 定义API端点
# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.get_json()
#     input_text = data['input']
#     output_text = generate_output(input_text)
#     return jsonify({'output': output_text})

# # 主页
# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)




# # encode
# parser = argparse.ArgumentParser()
# parser.add_argument('--input', default=None, type=str, help='ft input file')
# parser.add_argument('--vocab', type=str, default=None, help='vocab path')
# parser.add_argument('--output', default=None, type=str, help='ft output file')
# parser.add_argument('--add_bos', action='store_true', help='')
# parser.add_argument('--add_eos', action='store_true', help='')
# args = parser.parse_args()

# enc = encoder.get_encoder(args.vocab)

# # a替换成已经做好json格式的输入
# a = "{\"context\": \"name : Blue Spice | Type : coffee shop | area : city centre\", \"completion\": \"A coffee shop in the city centre area called Blue Spice .\"}"

# items = json.loads(a.strip())
# context = items['context']
# completion = items['completion']

# bos = 50256
# eos = 50256
# context_bpes, _ = enc.encode(context) 
# context_bpes += [bos] if args.add_bos else []

# completion_bpes, _ = enc.encode(' ' + completion)
# completion_bpes += [eos] if args.add_eos else []

# ft_json = {}
# ft_json['context'] = context_bpes
# ft_json['completion'] = completion_bpes 
# print(json.dumps(ft_json)+'\n')



# beam search
import argparse
import time
import math
import os, sys
import json
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.set_printoptions(threshold=100000)

import numpy as np



from exp_utils import create_exp_dir


from model import GPT2Config, GPT2LMModel


parser = argparse.ArgumentParser(description='PyTorch GPT2 beam decoding')


parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'],
                    help='model names')


def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx).contiguous().detach() for layer_past in past)


def _calc_banned_ngram_tokens(
    prev_input_ids: Tensor, 
    num_hypos: int, 
    no_repeat_ngram_size: int, 
    cur_len: int
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(
    lprobs, 
    batch_size, 
    num_beams, 
    prev_output_tokens, 
    repetition_penalty
):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """

    for i in range(batch_size * num_beams):
        print('prev_output_tokens.shape', prev_output_tokens.shape)
        print('prev_output_tokens[i].shape', prev_output_tokens[i].shape)

        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def _postprocess_next_token_scores(
    scores,
    history,
    cur_len,
    batch_size,
    num_beams,
    repetition_penalty=1.0,                                
    no_repeat_ngram_size=4,
    bad_words_ids=None,
    min_length=0,
    max_length=100,
    eos_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = _calc_banned_ngram_tokens(
                history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def _add_beam_candidate(
    best_score, 
    best_sequence, 
    batch_size, 
    num_beams, 
    beam_scores, 
    history, 
    eos_token_id=None
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.view(-1)[_i] / cur_len ** 0.8 # args.length_penalty=0.8

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


def beam(model, data_iter, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}
    with torch.no_grad():
        for idx, data in enumerate(data_iter):
            data = {key: value for key, value in data.items()}

            _id = data['id'].to("cuda")
            _query = data['query'].to("cuda")
            _query_len = data['query_len'].to("cuda")

            ## local adaptation start.

            ## local adaptation end.


            output = None
            score = None

            batch_size = _id.size(0)
            num_beams = 10 # args.beam
            length_penalty = 0.8 # args.length_penalty

            _batch = torch.arange(0, _id.size(0), device="cuda", dtype=torch.long)
            
            past = None
            len_past = None

            _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

            _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)
            
            # scores for each sentence in the beam
            beam_scores = torch.zeros(
                (batch_size, num_beams), dtype=torch.float, device=_query.device
            )

            best_sequence = torch.zeros(
                (batch_size, 64), dtype=torch.long, device=_query.device
            )
            best_score = {}

            history = None
            with torch.no_grad():
                for i in range(0, 64):# eval_len=64
                    if i == 0:
                        logits, past = model(_query) 
                        logits = logits[_bbatch, (_query_len-1).long(), :] # batch_size * beam, vocab
                    else:
                        #print('token_id.shape', token_id.shape, token_id)
                        #print('past.shape', past[0].shape)
                        #print('len_past.shape', len_past.shape, len_past)
                        
                        logits, past = model(token_id, past=past, len_past=len_past) 
                        logits = logits[:, -1, :]    # batch_size * beam, vocab

                    # logits = _postprocess_next_token_scores(           
                    #     logits,
                    #     history,
                    #     i,
                    #     batch_size,
                    #     num_beams,
                    #     repetition_penalty=args.repetition_penalty,                                
                    #     no_repeat_ngram_size=args.no_repeat_ngram_size,
                    #     min_length=args.min_length,
                    #     eos_token_id=args.eos_token_id,
                    # )
                        
                    logits = _postprocess_next_token_scores(           
                        logits,
                        history,
                        i,
                        batch_size,
                        num_beams,
                        repetition_penalty=1.0,                                
                        no_repeat_ngram_size=4,
                        eos_token_id=[50256, 628],
                    )

                    softmax_probs = F.softmax(logits, dim=-1)
                    ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

                    vocab_size = softmax_probs.shape[-1] 
                    

                    _logprob = torch.log(softmax_probs) # batch_size * beam, vocab
                    if i == 0:
                        next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :] # batch_size, vocab
                        
                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)
                        next_scores = next_scores.view(batch_size, -1) # batch_size, beam * vocab

                    next_scores, next_tokens = torch.topk(
                        next_scores, num_beams, dim=1, largest=True, sorted=True
                    )     # batch_size, num_beams
                    
                    beam_id = (next_tokens // vocab_size).view(-1)    # batch_size * num_beams
                    token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1) # batch_size, num_beams

                    beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
                    past = _reorder_cache(past, beam_idx.view(-1))                
                    beam_scores = next_scores # batch_size, num_beams
                    len_past = (_query_len + i).long()

                    if history is None:
                        history = token_id.detach()
                    else:
                        history = torch.cat((history[beam_idx.view(-1)], token_id.detach()), dim=1).detach()
                    
                    #--eos_token_id=628
                    _add_beam_candidate(
                        best_score, best_sequence, batch_size, num_beams, beam_scores, history, 
                        eos_token_id=[50256, 628]
                    )
                
                _add_beam_candidate(
                    best_score, best_sequence, batch_size, num_beams, beam_scores, history
                )


            with torch.no_grad():
                _id = _id.cpu()
                output = best_sequence.cpu()
                #score = distributed_gather(args, score)

            _id = _id.view(-1).cpu()
            output = output.view(-1, output.shape[-1]).cpu()
            #score = score.view(-1, score.shape[-1]).cpu()

            for _b in range(0, _id.shape[-1]):
                _i = int(_id[_b].item())
                all_predictions[_i] = {}
                all_predictions[_i]['id'] = _i
                all_predictions[_i]['predict'] = output[_b].tolist()
                #all_predictions[_i]['score'] = score[_b].tolist()

            if idx % 10 == 0:
                print('inference samples', idx)

   
    for _i in all_predictions:
        print(json.dumps(all_predictions[_i]) + '\n')
    

from torch.utils.data import Dataset
def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):

    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len
class FT_Dataset(Dataset):
    def __init__(self, ft_file, batch_size, max_seq_length, 
                 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0, 
                 prefix_cursor=1000000, infix_cursor=2000000):
        self.ft_file = ft_file
        self.ft_samples = self.read_ft_file(ft_file)
        self.batch_size = batch_size
        self.num_examples = len(self.ft_samples)
        self.max_seq_length = max_seq_length
        self.max_eval_length = max_eval_length
        self.rng = random.Random(911)
        self.joint_lm = joint_lm

        self.num_batches = int((self.num_examples + self.batch_size - 1) / self.batch_size) 

        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

    def __len__(self):
        return self.num_batches * self.batch_size
        
    def __getitem__(self, item):
        if(item >= self.num_examples):
            item = self.rng.randint(0, self.num_examples - 1)

        example = self.ft_samples[item]
        context = example[0]
        completion = example[1]

        pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)] 
        intokens = [i + self.infix_cursor for i in range(0, self.infix_len)] 

        conditions = pretokens + context + intokens 
        _input, _input_len = padding_tokens(conditions + completion, self.max_seq_length, 0, 1)

        pad_targets = [0 for i in range(0, self.prefix_len)] + context + [0 for i in range(0, self.infix_len)] + completion
        _target, _ = padding_tokens(pad_targets[1:], self.max_seq_length, 0, 1)

        if not self.joint_lm:
            _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        else:
            _msk = [1.0] * (_input_len - 1)

        _msk, _ = padding_tokens(_msk, self.max_seq_length, 0.0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        
        _query, _query_len = padding_tokens(
            conditions, self.max_seq_length, 0, -1, 
            max_context_length = self.max_seq_length - self.max_eval_length
        )
        output["query"] = torch.tensor(_query, dtype=torch.long)
        output["query_len"] = torch.tensor(_query_len, dtype=torch.long)

        output["input"] = torch.tensor(_input, dtype=torch.long) 
        output["target"] = torch.tensor(_target, dtype=torch.long) 

        output["mask"] = torch.tensor(_msk, dtype=torch.float)
        return output

    def read_ft_file(self, items):
        ft_samples = []
        context = items['context']
        completion = items['completion']
        ft_samples.append([context, completion])
        return ft_samples


if __name__ == '__main__':
    args = parser.parse_args()

    json_data = "{\"context\": [3672, 1058, 4518, 43537, 930, 5994, 1058, 6891, 6128, 930, 1989, 1058, 1748, 7372, 50256], \"completion\": []}"
    
    # json_data = "{'context': [3672, 1058, 4171, 50256], 'completion': [50256]} "
    data_dict = json.loads(json_data)
    batch_size = 1
    seq_len = 512
    eval_len = 64
    valid_data = FT_Dataset(
        data_dict, batch_size, seq_len, eval_len, 
    )    
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, num_workers=0, shuffle=False, 
        pin_memory=False, drop_last=False
    )

    init_checkpoint = ""
    lora_dim = 4
    lora_alpha = 32
    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=lora_dim, lora_attn_alpha=lora_alpha,
        )
        
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=lora_dim, lora_attn_alpha=lora_alpha,
        )
        init_checkpoint = "./trained_models/GPT2_M/e2e/model.35055.pt"
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=lora_dim, lora_attn_alpha=lora_alpha,
        )

    lm_net = GPT2LMModel(config)
    print('loading model pretrained weight.')
    cp = torch.load(init_checkpoint, map_location=torch.device('cpu'))
    lm_net.load_weight(cp)    
    lm_net = lm_net.cuda()

    print('model sampling ...')
    beam(lm_net, valid_loader, args)
    print('cleanup dist ...')








# # decode生成的句子
# import json
# import numpy as np
# import argparse
# import os
# import sys
# import re
# import json

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data

# import encoder


# parser = argparse.ArgumentParser()

# parser.add_argument('--vocab', type=str, default=None, help='vocab path')

# parser.add_argument('--sample_file', default=None, type=str, help='ft sample file')
# parser.add_argument('--input_file', default=None, type=str, help='ft input file')

# parser.add_argument('--output_ref_file', default=None, type=str, help='output reference file')
# parser.add_argument('--output_pred_file', default=None, type=str, help='output predicion file')

# parser.add_argument('--ref_unique_file', default=None, type=str, help='reference unique id file')

# parser.add_argument('--ref_type', default='e2e', choices=['e2e', 'webnlg', 'dart'], 
#                     help='e2e style reference type; webnlg style reference type.')
# parser.add_argument('--ref_num', default=4, type=int, help='number of references.')


# parser.add_argument('--tokenize', action='store_true', help='')
# parser.add_argument('--lower', action='store_true', help='')

# parser.add_argument('--filter', default='all', choices=['all', 'seen', 'unseen'], 
#                     help='for webnlg only, filter categories that are seen during training, unseen, or all')

# args = parser.parse_args()


# def stardard_tokenize(sent):
#     sent = ' '.join(re.split('(\W)', sent))
#     sent = sent.split()
#     sent = ' '.join(sent)
#     return sent


# def post_process(sent, is_tokenize, is_lower):
#     if is_lower:
#         sent = sent.lower()
#     if is_tokenize:
#         sent = stardard_tokenize(sent)

#     return sent


# if __name__ == "__main__":
#     enc = encoder.get_encoder(args.vocab)
#     with open(args.sample_file, 'r') as sample_reader, \
#         open(args.output_pred_file, 'w', encoding='utf8') as pred_writer:
#         for line in sample_reader:
#             items = json.loads(line.strip())
#             _pred_tokens = items['predict']
#             #assert _key in refer_dict
#             hypothesis = enc.decode(_pred_tokens).split('<|endoftext|>')[0].split('\n\n')[0].strip()
#         for hyp in hypothesis:
#             pred_writer.write(post_process(hyp, args.tokenize, args.lower))