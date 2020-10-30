from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import re
import random
import io

class Youtube_Transcript_DataLoader(Dataset):
    """
    Youtube dataset loader.
    Note: Use transcript as caption, for mask decoder pretrain task.
    """

    def __init__(
            self,
            csv,
            features_path,
            caption,
            tokenizer,
            min_time=10.0,
            features_path_3D=None,
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            max_words=30,
            min_words=0,
            n_pair=-1,          # -1 for test
            pad_token="[PAD]",
            max_frames=100,
            with_long_context=True,
            use_mil=False,
            only_sim=False,     # set automatically from model choice
            sampled_use_mil=False,
            pretrain_enhance_vmodal=False,
            video_dim=1024,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.features_path_2D = features_path
        self.features_path_3D = features_path_3D
        self.caption = caption
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.feature_framerate_3D = feature_framerate_3D
        self.max_words = max_words
        self.max_frames = max_frames
        self.min_words = min_words
        self.tokenizer = tokenizer
        self.n_pair = n_pair
        self.pad_token = pad_token
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = {'2d': features_path}
        if features_path_3D != '':
            self.feature_path['3d'] = features_path_3D
        self.with_long_context = with_long_context

        self.feature_size = video_dim

        self.only_sim = only_sim
        self.pretrain_enhance_vmodal = pretrain_enhance_vmodal
        self.iter_num = len(self.csv)

        self.use_mil = use_mil
        self.sampled_use_mil = sampled_use_mil
        if self.sampled_use_mil:        # sample from each video, has a high priority than use_mil.
            self.use_mil = True

        if self.use_mil:
            positive_n_pair = self.n_pair
            # Get iterator video ids
            video_id_list = [itm for itm in self.csv['video_id'].values]
            self.video_id2idx_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
            # Get all captions
            self.iter2video_pairs_dict = {}
            self.iter2video_pairslist_dict = {}
            iter_idx_mil_ = 0
            for video_id in video_id_list:
                caption = self.caption[video_id]
                n_caption = len(caption['start'])

                sub_list = []
                if self.n_pair < 0 or self.n_pair == 1:
                    for sub_id in range(n_caption):
                        sub_list.append([sub_id])
                else:
                    sb_ls_ = list(range(n_caption))
                    sb_st_ = set(sb_ls_)
                    if self.n_pair > n_caption:
                        sb_ls_ = sb_ls_ * (self.n_pair // n_caption + 1)
                        sb_ls_ = sb_ls_[:self.n_pair]
                        for sub_id in np.arange(0, len(sb_ls_), self.n_pair):
                            sub_list.append(sb_ls_[sub_id: sub_id + self.n_pair])
                    else:
                        sb_ls_ = sb_ls_ + sb_ls_[:(((n_caption+positive_n_pair-1)//positive_n_pair)*positive_n_pair-n_caption)]
                        for sub_id in np.arange(0, len(sb_ls_), positive_n_pair):
                            pos_ls = sb_ls_[sub_id: sub_id + positive_n_pair]
                            sub_list.append(pos_ls)

                for sub_e in sub_list:
                    self.iter2video_pairs_dict[iter_idx_mil_] = (video_id, sub_e)
                    iter_idx_mil_ += 1
                self.iter2video_pairslist_dict[video_id] = sub_list

        if self.use_mil and self.sampled_use_mil is False:
            self.iter_num = len(self.iter2video_pairs_dict)

    def __len__(self):
        return self.iter_num

    def _mask_tokens(self, words, orig2token_tuple_list=None, chunk_positions=None):
        token_labels = []
        masked_tokens = words.copy()

        if chunk_positions is not None and len(chunk_positions)>0:
            token_labels = [-1] * len(masked_tokens)
            for chunk_ind in chunk_positions:
                start_, len_ = orig2token_tuple_list[chunk_ind]
                if random.random() < 0.15:
                    token_labels[start_:start_+len_] = [self.tokenizer.vocab[tk_] for tk_ in masked_tokens[start_:start_+len_]]
                    masked_tokens[start_:start_+len_] = ["[MASK]"] * len_
        else:
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                else:
                    token_labels.append(-1)

        return masked_tokens, token_labels

    def _get_text(self, video_id, n_pair_max, sub_ids=None, only_sim=False, enhance_vmodel=False):
        caption = self.caption[video_id]

        if self.use_mil:
            k = len(sub_ids)
            r_ind = sub_ids
        else:
            n_caption = len(caption['start'])
            if n_pair_max == -1:
                k = n_caption
                r_ind = range(n_caption)
            else:
                k = n_pair_max
                if k <= n_caption:
                    r_ind = np.random.choice(range(n_caption), k, replace=False)
                else:
                    r_ind_must = np.array(range(n_caption))
                    r_ind_rand = np.random.choice(range(n_caption), k-n_caption, replace=True)
                    r_ind = np.concatenate((r_ind_must, r_ind_rand), axis=0)
                np.random.shuffle(r_ind)

        starts = np.zeros(k)
        ends = np.zeros(k)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            words, start_, end_ = self._get_single_transcript(caption, ind, with_long_context=self.with_long_context)
            caption_words = words.copy()
            starts[i], ends[i] = start_, end_

            orig2token_tuple_list, chunk_positions = None, None
            # # For entity mask
            # orig_sentence, orig_to_token_map = generate_org_sentence_from_tokens(words.copy())
            # four_tuples_ = generate_knoledge_candidate(orig_sentence, orig_to_token_map)
            # _, orig2token_tuple_list, chunk_positions, _ = four_tuples_

            if enhance_vmodel:
                words = []      # mask all input text

            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            if only_sim is False:
                # # For entity mask : +1 due to [CLS]
                # orig2token_tuple_list = [(s_+1, len_) for s_, len_ in orig2token_tuple_list if s_+len_ <= total_length_with_CLS]
                # chunk_positions = [ind for ind in chunk_positions if ind<len(orig2token_tuple_list)]

                # For generate captions
                if len(caption_words) > total_length_with_CLS:
                    caption_words = caption_words[:total_length_with_CLS]
                input_caption_words = ["[CLS]"] + caption_words
                output_caption_words = caption_words + ["[SEP]"]

                masked_tokens, token_labels = self._mask_tokens(words, orig2token_tuple_list, chunk_positions)
                masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                masked_input_caption_words, input_token_labels = self._mask_tokens(input_caption_words)
                input_caption_words = masked_input_caption_words.copy()

                while len(masked_token_ids) < self.max_words:
                    masked_token_ids.append(0)
                    token_labels.append(-1)
                assert len(masked_token_ids) == self.max_words
                assert len(token_labels) == self.max_words

                # For generate captions
                input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
                output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
                decoder_mask = [1] * len(input_caption_ids)
                while len(input_caption_ids) < self.max_words:
                    input_caption_ids.append(0)
                    output_caption_ids.append(0)
                    decoder_mask.append(0)
                assert len(input_caption_ids) == self.max_words
                assert len(output_caption_ids) == self.max_words
                assert len(decoder_mask) == self.max_words

                pairs_masked_text[i] = np.array(masked_token_ids)
                pairs_token_labels[i] = np.array(token_labels)

                pairs_input_caption_ids[i] = np.array(input_caption_ids)
                pairs_output_caption_ids[i] = np.array(output_caption_ids)
                pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends

    def _get_single_transcript(self, caption, ind, with_long_context=True):
        start, end = ind, ind
        words = self.tokenizer.tokenize(str(caption['text'][ind]))
        diff = caption['end'][end] - caption['start'][start]
        while with_long_context and (len(words) < self.min_words or diff < self.min_time):
            if start > 0 and end < len(caption['end']) - 1:
                next_words = self.tokenizer.tokenize(str(caption['text'][end + 1]))
                prev_words = self.tokenizer.tokenize(str(caption['text'][start - 1]))
                d1 = caption['end'][end + 1] - caption['start'][start]
                d2 = caption['end'][end] - caption['start'][start - 1]
                if (self.min_time > 0 and d2 <= d1) or \
                    (self.min_time == 0 and len(next_words) <= len(prev_words)):
                    start -= 1
                    words = prev_words + words
                else:
                    end += 1
                    words.extend(next_words)
            elif start > 0:
                words = self.tokenizer.tokenize(str(caption['text'][start - 1])) + words
                start -= 1
            elif end < len(caption['end']) - 1:
                words.extend(self.tokenizer.tokenize(str(caption['text'][end + 1])))
                end += 1
            else:
                break
            diff = caption['end'][end] - caption['start'][start]
        return words, caption['start'][start], caption['end'][end]

    def _expand_video_slice(self, s, e, si, ei, fps, video_features, fps_k):
        start = int(s[si] * fps)
        end = int(e[ei] * fps) + 1

        if start > end:
            start, end = end, start
        video_slice = video_features[start:end]

        expand_left = True
        while len(video_slice) < 1:
            if si==0 and ei==len(s)-1:
                break
            if expand_left:
                expand_left = False
                si = si-1 if si>0 else si
            else:
                expand_left = True
                ei = ei+1 if ei<len(e)-1 else ei
            start = int(s[si] * fps)
            end = int(e[ei] * fps) + 1
            if start > end:
                start, end = end, start
            video_slice = video_features[start:end]

        # # Note: to alignment the features between 2d and 3d, due to the fps is different.
        # if fps_k == "2d":
        #     indices = sorted([idx for idx in range(len(video_slice)) if idx % 2 == 1]
        #                      + [idx for idx in range(len(video_slice))])
        #     video_slice = np.take(video_slice, indices, axis=0)

        if self.max_frames < video_slice.shape[0]:
            video_slice = video_slice[:self.max_frames]

        return video_slice, start, end

    def _get_video(self, idx, s, e, only_sim=False):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)

        max_video_length = [0] * len(s)
        f_title = "feature_file_2D"
        fps_k = "2d"

        video = np.zeros((len(s), self.max_frames, self.feature_size), dtype=np.float)
        feature_file = os.path.join(self.feature_path[fps_k], self.csv[f_title].values[idx])
        try:
            video_features = np.load(feature_file)

            for i in range(len(s)):
                # start = int(s[i] * self.fps[fps_k])
                # end = int(e[i] * self.fps[fps_k]) + 1
                # video_slice = video_features[start:end]
                #
                # if self.max_frames < video_slice.shape[0]:
                #     video_slice = video_slice[:self.max_frames]
                if len(video_features) < 1:
                    raise ValueError("{} is empty.".format(feature_file))
                video_slice, start, end = self._expand_video_slice(s, e, i, i,
                                                                   self.fps[fps_k], video_features, fps_k)
                slice_shape = video_slice.shape
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
                if len(video_slice) < 1:
                    pass
                    # video_slice = video_features
                    # if self.max_frames < video_slice.shape[0]:
                    #     video_slice = video_slice[:self.max_frames]
                else:
                    video[i][:slice_shape[0]] = video_slice
        except Exception as e:
            print("video_id: {} error.".format(feature_file))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(s))]
        masked_video = video.copy()
        if only_sim is False:
            for i, video_pair_ in enumerate(masked_video):
                for j, _ in enumerate(video_pair_):
                    if j < max_video_length[i]:
                        prob = random.random()
                        # mask token with 15% probability
                        if prob < 0.15:
                            masked_video[i][j] = [0.] * video.shape[-1]
                            video_labels_index[i].append(j)
                        else:
                            video_labels_index[i].append(-1)
                    else:
                        video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def second_to_stamp(self, in_seconds):
        m, s = divmod(in_seconds, 60)
        h, m2 = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m2, s)

    def __getitem__(self, feature_idx):
        if self.sampled_use_mil:  # sample from each video, has a high priority than use_mil.
            idx = feature_idx
            video_id = self.csv['video_id'].values[idx]
            sub_list = self.iter2video_pairslist_dict[video_id]
            ranint = np.random.randint(0, len(sub_list))
            sub_ids = sub_list[ranint]
        elif self.use_mil:
            video_id, sub_ids = self.iter2video_pairs_dict[feature_idx]
            idx = self.video_id2idx_dict[video_id]
        else:
            idx = feature_idx
            video_id = self.csv['video_id'].values[idx]
            sub_ids = None

        enhance_vmodel = False
        if self.only_sim is False and self.pretrain_enhance_vmodal:
            prob = random.random()
            if prob < 0.15:     # mask all text by rate 0.15
                enhance_vmodel = True

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
        pairs_decoder_mask, pairs_output_caption_ids, \
        starts, ends = self._get_text(video_id, self.n_pair, sub_ids, only_sim=self.only_sim, enhance_vmodel=enhance_vmodel)

        video, video_mask, masked_video, video_labels_index = self._get_video(idx, starts, ends, only_sim=self.only_sim)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids
