# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np

import torch
from torch import nn
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'decoder_config.json'
WEIGHTS_NAME = 'decoder_pytorch_model.bin'


class DecoderConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `DecoderModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 max_target_embeddings=128,
                 num_decoder_layers=1):
        """Constructs DecoderConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `DecoderModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `DecoderModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            max_target_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            num_decoder_layers:
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.max_target_embeddings = max_target_embeddings
            self.num_decoder_layers = num_decoder_layers
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, decoder_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(decoder_model_embedding_weights.size(1),
                                 decoder_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = decoder_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(decoder_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, decoder_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, decoder_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(ACT2FN["gelu"](self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class DecoderAttention(nn.Module):
    def __init__(self, config):
        super(DecoderAttention, self).__init__()
        self.att = MultiHeadAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q, k, v, attention_mask):
        att_output, attention_probs = self.att(q, k, v, attention_mask)
        attention_output = self.output(att_output, q)
        return attention_output, attention_probs

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.slf_attn = DecoderAttention(config)
        self.enc_attn = DecoderAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        slf_output, _ = self.slf_attn(dec_input, dec_input, dec_input, slf_attn_mask)
        dec_output, dec_att_scores = self.enc_attn(slf_output, enc_output, enc_output, dec_enc_attn_mask)
        intermediate_output = self.intermediate(dec_output)
        dec_output = self.output(intermediate_output, dec_output)
        return dec_output, dec_att_scores

class DecoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, decoder_word_embeddings_weight, decoder_position_embeddings_weight):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_target_embeddings, config.hidden_size)
        self.word_embeddings.weight = decoder_word_embeddings_weight
        self.position_embeddings.weight = decoder_position_embeddings_weight

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        layer = DecoderLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_decoder_layers)])

    def forward(self, hidden_states, encoder_outs, self_attn_mask, attention_mask, output_all_encoded_layers=False):
        dec_att_scores = None
        all_encoder_layers = []
        all_dec_att_probs = []
        for layer_module in self.layer:
            hidden_states, dec_att_scores = layer_module(hidden_states, encoder_outs, self_attn_mask, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_dec_att_probs.append(dec_att_scores)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_dec_att_probs.append(dec_att_scores)
        return all_encoder_layers, all_dec_att_probs

class DecoderClassifier(nn.Module):
    def __init__(self, config, embedding_weights):
        super(DecoderClassifier, self).__init__()
        self.cls = BertOnlyMLMHead(config, embedding_weights)

    def forward(self, hidden_states):
        cls_scores = self.cls(hidden_states)
        return cls_scores

class DecoderModel(PreTrainedModel):

    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, config, decoder_word_embeddings_weight, decoder_position_embeddings_weight):
        super(DecoderModel, self).__init__(config)
        self.config = config
        self.max_target_length = config.max_target_embeddings
        self.embeddings = DecoderEmbeddings(config, decoder_word_embeddings_weight, decoder_position_embeddings_weight)
        self.decoder = Decoder(config)
        self.classifier = DecoderClassifier(config, decoder_word_embeddings_weight)
        self.apply(self.init_weights)

    def forward(self, input_ids, encoder_outs=None, answer_mask=None, encoder_mask=None):
        """
        Args:
            input_ids (LongTensor): previous decoder outputs of shape `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_outs (Tensor, optional): output from the encoder, used for encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape `(batch, tgt_len, src_len)`
        """
        embedding_output = self.embeddings(input_ids)

        extended_encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)   # b x 1 x 1 x ls
        extended_encoder_mask = extended_encoder_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_encoder_mask = (1.0 - extended_encoder_mask) * -10000.0

        extended_answer_mask = answer_mask.unsqueeze(1).unsqueeze(2)
        extended_answer_mask = extended_answer_mask.to(dtype=self.dtype)  # fp16 compatibility

        sz_b, len_s, _ = embedding_output.size()
        subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=embedding_output.device, dtype=embedding_output.dtype), diagonal=1)
        self_attn_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1).unsqueeze(1)  # b x 1 x ls x ls
        slf_attn_mask = ((1.0 - extended_answer_mask) + self_attn_mask).gt(0).to(dtype=self.dtype)
        self_attn_mask = slf_attn_mask * -10000.0

        decoded_layers, dec_att_scores = self.decoder(embedding_output,
                                      encoder_outs,
                                      self_attn_mask,
                                      extended_encoder_mask,
                                      )
        sequence_output = decoded_layers[-1]
        cls_scores = self.classifier(sequence_output)

        return cls_scores
