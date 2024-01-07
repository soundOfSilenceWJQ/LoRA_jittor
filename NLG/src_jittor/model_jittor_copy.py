#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import copy
import math

import jittor as jt
from jittor import nn
# import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss, MSELoss
# import torch.nn.functional as F
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LambdaLR
# from torch.nn.parameter import Parameter

import loralib_jittor as lora


def gelu(x):
    return 0.5 * x * (1 + jt.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * jt.pow(x, 3))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + jt.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jt.pow(x, 3.0))))


def swish(x):
    return x * jt.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + jt.erf(x / math.sqrt(2.0)))


# class LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root)."""
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
    
# import jittor as jt

class LayerNorm(jt.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = jt.Var(jt.ones([hidden_size]))
        self.bias = jt.Var(jt.zeros([hidden_size]))
        self.variance_epsilon = eps

    def execute(self, x):
        u = x.mean(-1, keepdims=True)
        s = (x - u).pow(2).mean(-1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



# class Conv1D(nn.Module):
#     def __init__(self, nf, nx):
#         super(Conv1D, self).__init__()
#         self.nf = nf
#         w = torch.empty(nx, nf)
#         nn.init.normal_(w, std=0.02)
#         self.weight = Parameter(w)
#         self.bias = Parameter(torch.zeros(nf))

#     def forward(self, x):
#         size_out = x.size()[:-1] + (self.nf,)
#         x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
#         x = x.view(*size_out)
#         return x
# class Conv1D(jt.Module):
#     def __init__(self, nf, nx):
#         super(Conv1D, self).__init__()
#         self.nf = nf
#         w = jt.randn([nx, nf]) * 0.02
#         self.weight = jt.Var(w)
#         self.bias = jt.Var(jt.zeros(nf))

#     def execute(self, x):
#         size_out = x.shape[:-1] + (self.nf,)
#         x = jt.matmul(x.view(-1, x.shape[-1]), self.weight) + self.bias
#         x = x.view(*size_out)
#         return x
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = jt.normal(0,0.02,size=(nx, nf))
        self.bias = jt.zeros(nf)

    def execute(self, x):
        shape_out = x.shape[:-1] + (self.nf,)
        x = jt.matmul(x.reshape(-1, x.shape[-1]), self.weight) + self.bias
        x = x.reshape(shape_out)
        return x



class Attention(jt.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        
        assert n_state % config.n_head == 0
        # self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.bias = jt.tril(jt.ones((1, 1, n_ctx, n_ctx))).view(1, 1, n_ctx, n_ctx)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = lora.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config
    
    def _attn(self, q, k, v, len_kv=None):
        # w = torch.matmul(q, k)
        w = jt.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            # _len = torch.arange(k.size(-1), device=k.device)
            _len = jt.arange(k.size(-1))
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = jt.nn.softmax(w, dim=-1)
        return jt.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def execute(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = jt.cat((past_key, key), dim=-1)
                value = jt.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = jt.arange(0, key.shape[0], dtype=jt.int64)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = jt.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present
    


class MLP(jt.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def execute(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(jt.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def execute(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(jt.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = jt.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = jt.nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = jt.nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config


    def execute(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = jt.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=jt.int64
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2LMHead(jt.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = jt.nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def execute(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        lora_r_dropout=0.0,
        fix_dropout=0.0,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.fix_dropout = fix_dropout


class GPT2LMModel(jt.Module):
    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self._init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def execute(
        self, 
        input_ids, 
        lm_labels=None, 
        lm_mask=None, 
        past=None, 
        len_past=None, 
        label_smooth=0.0,
        is_report_accuracy=False
    ):
        _batch, _len = input_ids.shape
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:

            if is_report_accuracy:
                _pred_token = jt.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask

                # _t1_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                # _all_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                _t1_acc = jt.zeros(_batch, dtype=jt.float32)
                _all_acc = jt.zeros(_batch, dtype=jt.float32)

                
                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break  

                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0

                #_t1_acc = _t1_acc * 1.0 / _batch
                #_all_acc = _all_acc * 1.0 / _batch

            if label_smooth > 0.0001:
                logprobs = jt.nn.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss = jt.nn.cross_entropy_loss(output=lm_logits.view(-1, lm_logits.size(-1)), 
                                                    target=lm_labels.view(-1), ignore_index=-1, reduction='none').view(_batch, _len)
                # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

            if lm_mask is None:
                lm_mask = jt.ones(loss.shape, dtype=loss.dtype)
                # lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask 

            loss = loss.sum() / (lm_mask.sum() + 0.0001)

            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss
        return lm_logits, presents
           
    def _init_weights(self, module):
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

        if isinstance(module, (jt.nn.Linear, jt.nn.Embedding)):
            module.weight.data = jt.randn(module.weight.shape) * 0.02
        # if module.bias is not None:
        #     module.bias.data.zero_()
        elif isinstance(module, jt.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # if module.bias is not None:
            #     module.bias.data.zero_()
        if isinstance(module, jt.nn.Linear) and module.bias is not None:
            module.bias.data.fill(0)


    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer.load_state_dict(state_dict, strict=False)
        self.set_tied()
