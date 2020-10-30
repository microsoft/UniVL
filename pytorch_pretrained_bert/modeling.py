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
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.bert_module import PreTrainedBertModel, BertModel, BertLayerNorm, BertOnlyMLMHead
from pytorch_pretrained_bert.visual_module import PreTrainedVisualBertModel, VisualBertModel, VisualBertLayerNorm, VisualBertOnlyMLMHead
from pytorch_pretrained_bert.cross_module import PreTrainedCrossBertModel, CrossBertModel, CrossBertLayerNorm
from pytorch_pretrained_bert.decoder_module import PreTrainedDecoderBertModel, DecoderBertModel, DecoderBertLayerNorm

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class LayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.decoder = None

    def init_bert_weights(self, module):
        """ Initialize the weights.
                """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, BertLayerNorm) \
                or isinstance(module, LayerNorm) \
                or isinstance(module, VisualBertLayerNorm) \
                or isinstance(module, CrossBertLayerNorm) \
                or isinstance(module, DecoderBertLayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise ValueError("`TODO: Just bert has this function, call it at here.`")

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix+key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-"*20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name, pretrained_visual_model_name, pretrained_cross_model_name,
                        pretrained_decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = PreTrainedBertModel.get_config(pretrained_model_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        # visual_state_dict is no use here
        visual_config, _ = PreTrainedVisualBertModel.get_config(pretrained_visual_model_name, cache_dir, state_dict=None, task_config=task_config)
        # visual_state_dict is no use here
        cross_config, _ = PreTrainedCrossBertModel.get_config(pretrained_cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = PreTrainedDecoderBertModel.get_config(pretrained_decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        # Instantiate model.
        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)

        assert model.bert is not None
        assert model.visual is not None
        # assert model.cross is not None
        # assert model.decoder is not None

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)      # video * text

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=torch.uint8)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

class VLBert(PreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config):
        super(VLBert, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._choice_sim = True
        self._cross_model = False

        if hasattr(self.task_config, 'cross_model') and self.task_config.cross_model:
            self._choice_sim = False
            self._cross_model = self.task_config.cross_model

        self.train_sim_after_cross = False
        if self._choice_sim and hasattr(self.task_config, 'train_sim_after_cross') and self.task_config.train_sim_after_cross:
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        self.with_sim_in_decoder = True
        if hasattr(self.task_config, 'without_sim_in_decoder') and self.task_config.without_sim_in_decoder:
            self.with_sim_in_decoder = False
            show_log(task_config, "Set no sim after cross.")

        self.pretrain_without_decoder = False
        if hasattr(self.task_config, 'pretrain_without_decoder') and self.task_config.pretrain_without_decoder:
            self.pretrain_without_decoder = True
            show_log(task_config, "Set pretrain without decoder.")

        show_log(task_config, "sim:{}, cross_model:{}".format(self._choice_sim, self._cross_model))

        # The name should be `bert`, if dynamic vocabulary is needed
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight

        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualBertModel(visual_config)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight

        if self._choice_sim is False or self.train_sim_after_cross:
            self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
            self.cls_visual = VisualBertOnlyMLMHead(visual_config, visual_word_embeddings_weight)

            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossBertModel(cross_config)

            if self.train_sim_after_cross is False:
                if self.task_config.do_pretrain and self.pretrain_without_decoder:
                    show_log(task_config, "Pretrain without decoder.")
                else:
                    decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                               self.task_config, "decoder_num_hidden_layers")
                    self.decoder = DecoderBertModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)

        self.normalize_video = NormalizeVideo(task_config)

        mILNCELoss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        maxMarginRankingLoss = MaxMarginRankingLoss(margin=task_config.margin,
                                                    negative_weighting=task_config.negative_weighting,
                                                    batch_size=task_config.batch_size // task_config.n_gpu,
                                                    n_pair=task_config.n_pair,
                                                    hard_negative_rate=task_config.hard_negative_rate, )
        if task_config.use_mil:
            self.loss_fct = mILNCELoss
            show_log(task_config, "Using MILNCE.")
        else:
            self.loss_fct = maxMarginRankingLoss
            show_log(task_config, "Using Ranking Loss.")

        if self._cross_model:
            self.loss_fct = CrossEn()
            show_log(task_config, "Use CE Loss.")

        if self._choice_sim is False or self.train_sim_after_cross:
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)

            self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.pretrain_with_joint_sim = False
        if hasattr(self.task_config, 'pretrain_with_joint_sim'):
            if self.task_config.pretrain_with_joint_sim:
                self.pretrain_with_joint_sim = True
                show_log(task_config, "Set pretrain with joint embedding.")
                if task_config.use_mil:
                    self._pretrain_sim_loss_fct = mILNCELoss
                    show_log(task_config, "Pretrain with joint embedding using MILNCE.")
                else:
                    self._pretrain_sim_loss_fct = maxMarginRankingLoss
                    show_log(task_config, "Pretrain with joint embedding using Ranking Loss.")

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = self.normalize_video(video)

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids,
                                                                         attention_mask, video, video_mask, shaped=True)

        if self.training:
            loss = 0.
            if self._choice_sim:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss = loss + sim_loss

            if self._cross_model and pairs_masked_text is not None and pairs_token_labels is not None:

                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_output(pairs_masked_text, token_type_ids, attention_mask, masked_video, video_mask, shaped=True)

                    cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output_alm, visual_output_alm, attention_mask, video_mask)
                    sequence_cross_output, visual_cross_output = torch.split(cross_output, [attention_mask.size(-1), video_mask.size(-1)], dim=1)

                    alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)  # For mlm
                    loss = loss + alm_loss

                    nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)  # For mfm
                    loss += nce_loss

                    if self.pretrain_with_joint_sim:
                        sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                                shaped=True, _pretrain_joint=True)
                        sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                        loss = loss + sim_loss_joint

                if (input_caption_ids is not None) and \
                        ((self.task_config.do_pretrain and self.pretrain_without_decoder is False)
                         or (self.task_config.do_pretrain is False and self.task_config.task_type == "caption")):
                    if self.task_config.do_pretrain:
                        decoder_scores, res_tuples = self._get_decoder_score(sequence_output_alm, visual_output_alm,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    elif self.task_config.task_type == "caption":
                        decoder_scores, res_tuples = self._get_decoder_score(sequence_output, visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)

                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                    loss = loss + decoder_loss

                if (self.with_sim_in_decoder and self.task_config.do_pretrain) or \
                        self.task_config.task_type == "retrieval":
                    if self.task_config.do_pretrain:
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output_alm, visual_output_alm, attention_mask,
                                                                            video_mask, shaped=True)
                    elif self.task_config.task_type == "retrieval":
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                                            video_mask, shaped=True)
                    sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
                    loss = loss + sim_loss_text_visual

            return loss
        else:
            return None

    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

    # For inference
    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        encoded_layers, _ = self.bert("bi", input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        visual_layers, _ = self.visual("bi", video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]

        # cannot add any cross code here
        # beacause sequence and visual features are exclusive

        return sequence_output, visual_output

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    # For inference
    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if self._cross_model and _pretrain_joint is False:
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            if self.train_sim_after_cross:
                retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
            else:
                text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
                # Do a cosine simlarity
                if self.task_config.use_mil is False:
                    text_out = F.normalize(text_out, dim=-1)
                    video_out = F.normalize(video_out, dim=-1)
                retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        # Generate one pair (x, y) ===>
        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)
        # <===
        cross_layers, pooled_output = self.cross("bi", concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _get_decoder_score(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, shaped=False):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        decoder_scores = self.decoder(input_caption_ids, input_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask)

        return decoder_scores, res_tuples

    def decoder_caption(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        """
        :param get_logits: Used in Beam Search
        :return:
        """
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        assert self.pretrain_without_decoder is False
        decoder_scores, _ = self._get_decoder_score(sequence_output, visual_output,
                                                 input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result