import jittor as jt
from jittor import nn

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(jt.zeros((r, num_embeddings), dtype=self.weight.dtype))
            self.lora_B = nn.Parameter(jt.zeros((embedding_dim, r), dtype=self.weight.dtype))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight = self.weight.stop_grad()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight)
        if self.padding_idx is not None:
            with jt.no_grad():
                self.weight[self.padding_idx].assign(0)
        # nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.init.zero_(self.lora_A)
            jt.init.trunc_normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight -= jt.transpose(jt.matmul(self.lora_B, self.lora_A), 0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight += jt.transpose(jt.matmul(self.lora_B, self.lora_A), 0, 1) * self.scaling
                self.merged = True
        
    def execute(self, x: jt.Var):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.execute(self, x)
            after_A = nn.embedding(
                x, jt.transpose(self.lora_A, 0, 1)
            )
            result += jt.matmul(after_A, jt.transpose(self.lora_B, 0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.execute(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(jt.zeros((r, in_features), dtype=self.weight.dtype))
            self.lora_B = nn.Parameter(jt.zeros((out_features, r), dtype=self.weight.dtype))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight = self.weight.stop_grad()
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight = jt.transpose(self.weight, 0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            jt.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.init.zero_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight -= T(jt.matmul(self.lora_B, self.lora_A)) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight += T(jt.matmul(self.lora_B, self.lora_A)) * self.scaling
                self.merged = True       

    def execute(self, x: jt.Var):
        def T(w):
            return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = nn.linear(x, T(self.weight), bias=self.bias)            
            result += (jt.matmul(jt.matmul(self.lora_dropout(x), jt.transpose(self.lora_A, 0, 1)), jt.transpose(self.lora_B, 0, 1))) * self.scaling
            return result
        else:
            return nn.linear(x, T(self.weight), bias=self.bias)

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        # enable_lora = jt.array(enable_lora, dtype='bool')   # 将 enable_lora 转换为 jt.Var，
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                # jt.zeros((r * sum(enable_lora), in_features), dtype=self.weight.dtype))
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                # jt.zeros((out_features // len(enable_lora) * sum(enable_lora), r), dtype=self.weight.dtype)
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight = self.weight.stop_grad()
            # Compute the indices
            #TODO 这里到底是reshape还是view??也许都可以
            self.lora_ind = jt.zeros((out_features,), dtype='bool').reshape(len(enable_lora), -1)
            self.lora_ind[jt.array(enable_lora), :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight = jt.transpose(self.weight, 0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.init.zero_(self.lora_B)

    def zero_pad(self, x):
        result = jt.zeros((len(self.lora_ind), *x.shape[1:]), dtype=x.dtype)
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        #TODO conv1d到底怎么用？？
        a = self.lora_A.unsqueeze(0)
        # a: [batch_size, in_channel, seq_len]
        b = self.lora_B.unsqueeze(-1)
        # b: [out_channel, in_channel, kernel_size]
        # in_channels = a.shape[1]
        # out_channels = self.lora_B.shape[0]
        conv = nn.Conv1d(
            in_channels=a.shape[1], 
            out_channels=b.shape[0], 
            kernel_size=1,
            groups=sum(self.enable_lora),
            bias=False
        )
        conv.weight = b
        delta_w = conv(a).squeeze(0)
        return T(self.zero_pad(delta_w))
    # def merge_AB(self):
    #     def T(w):
    #         return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
    #     #TODO conv1d到底怎么用？？
    #     a = self.lora_A.unsqueeze(0)
    #     b = self.lora_B.unsqueeze(-1)
    #     in_channels = self.lora_A.shape[0]
    #     out_channels = self.lora_B.shape[0]
    #     kernel_size = 1
    #     stride = 1
    #     self.conv1d = nn.Conv1d(
    #         in_channels=in_channels, 
    #         out_channels=out_channels, 
    #         kernel_size=1,
    #         groups=sum(self.enable_lora)
    #     )
    #     # self.weight = self.lora_B.unsqueeze(-1)
    #     self.weight = self.lora_B.unsqueeze(-1).copy()
    #     delta_w = self.conv1d(self.lora_A.unsqueeze(0)).squeeze(0)
    #     s_tem = delta_w.shape
    #     delta_tem = delta_w
    #     return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight += self.merge_AB() * self.scaling
                self.merged = True        

    def execute(self, x: jt.Var):
        def T(w):
            return jt.transpose(w, 0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return nn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = nn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += jt.matmul(self.lora_dropout(x), T(jt.transpose(self.merge_AB()))) * self.scaling
            a_tem = 0
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                jt.zeros((r * kernel_size, in_channels * kernel_size), dtype=self.conv.weight.dtype)
            )
            self.lora_B = nn.Parameter(
              jt.zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size), dtype=self.conv.weight.dtype)
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight = self.conv.weight.stop_grad()
        self.reset_parameters()
        self.merged = False

    # TODO 这里不知道对不对
    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor.shape(1)
        num_output_fmaps = tensor.shape(0)
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = jt.prod(tensor.shape()[2:])

        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def reset_parameters(self):
        # self.conv.reset_parameters()
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out(self.conv.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.conv.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.init.zero_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight -= jt.matmul(self.lora_B, self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight += jt.matmul(self.lora_B, self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        out_channels = weight.shape[0]

        if groups == 1:
            N, C, L = x.shape
            Kw = weight.shape[-1]
            ol = (L + padding * 2 - Kw * dilation + dilation - 1) // stride + 1
            with jt.flag_scope(amp_reg=jt.flags.amp_reg | 36):
                xx = x.reindex([N, out_channels, C, ol, Kw], [
                    'i0',  # Nid
                    'i2',  # Cid
                    f'i3*{stride}-{padding}+i4*{dilation}',  # Lid+KWid
                ])
                ww = weight.broadcast(xx.shape, [0, 3, 4])
                yy = xx * ww
                y = yy.sum([2, 4])  # Kc, Kw
            if bias is not None:
                b = bias.broadcast(y.shape, [0, 2])
                y = y + b
            return y
        else:
            N, C, L = x.shape
            Kw = weight.shape[-1]
            G = groups
            CpG = C // G  # channels per group
            oc = out_channels
            ol = (L + padding * 2 - Kw * dilation + dilation - 1) // stride + 1
            xx = x.reindex([N, G, oc // G, CpG, ol, Kw], [
                'i0',  # Nid
                f'i1*{CpG}+i3',  # Gid
                f'i4*{stride}-{padding}+i5*{dilation}',  # Lid+KWid
            ])
            xx.compile_options = {"G": G}
            ww = weight.reindex([N, G, oc // G, CpG, ol, Kw], [
                f'i1*{oc // G}+i2',
                'i3',
                'i5',
            ])
            yy = xx * ww
            y = yy.reindex_reduce('add', [N, oc, ol], [
                'i0',
                f'i1*{oc // G}+i2',
                'i4',
            ])
            if bias is not None:
                b = bias.broadcast(y.shape, [0, 2])
                y = y + b
            return y

    def execute(self, x):
        if self.r > 0 and not self.merged:
            # return self.conv.conv2d( 
            #     x, 
            #     self.conv.weight + jt.matmul(self.lora_B, self.lora_A).view(self.conv.weight.shape) * self.scaling,
            #     self.conv.bias
            # )
            # TODO 这里肯定不对
            if isinstance(self.conv, nn.Conv1d):
                # 处理 Conv1d 的情况
                return self.conv1d(x, self.conv.weight + jt.matmul(self.lora_B, self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias)
            elif isinstance(self.conv, nn.Conv2d):
                # 处理 Conv2d 的情况
                return nn.conv2d(x, self.conv.weight + jt.matmul(self.lora_B, self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias)
            elif isinstance(self.conv, nn.Conv3d):
                # 处理 Conv3d 的情况
                return nn.conv3d(x, self.conv.weight + jt.matmul(self.lora_B, self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias)
            else:
                # 处理其他情况或引发异常
                raise ValueError("不支持的卷积层类型")
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# 可以扩展到其他类似的方式
class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)