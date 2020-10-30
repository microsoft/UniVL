from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (RandomSampler, SequentialSampler, TensorDataset)

from data_prefetch_unitl import DataLoaderX as DataLoader       # Enhanced Loader

import numpy as np
import random
import os
from collections import OrderedDict
from youcook_transcript_nopair_dataloader import Youcook_Transcript_NoPair_DataLoader
from youtube_transcript_dataloader import Youtube_Transcript_DataLoader
from rouge import Rouge
from nlgeval import compute_metrics, NLGEval
import pickle
import logging
import time
import argparse
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import VLBert
from pytorch_pretrained_bert.optimization_bert import BertAdam
from pytorch_pretrained_bert.beam import Beam

torch.distributed.init_process_group(backend="nccl")

global amp_pck_loaded_
try:
    from apex import amp
    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    amp.register_half_function(torch, 'einsum')
    amp_pck_loaded_ = True
except ImportError:
    amp_pck_loaded_ = False

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

global logger

def get_args(description='Youtube-Text-Video'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--train_csv', type=str, default='data/HowTo100M_v1.csv', help='train csv')
    parser.add_argument('--features_path_2D', type=str, default='feature_2d', help='feature path for 2D features')
    parser.add_argument('--features_path_3D', type=str, default='feature_3d', help='feature path for 3D features')
    parser.add_argument('--caption_path', type=str, default='data/caption.pickle', help='caption pickle file path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--min_words', type=int, default=0, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument('--youcook', type=int, default=0, help='Train on YouCook2 data')
    parser.add_argument('--eval_youcook', type=int, default=0, help='Evaluate on YouCook2 data')
    parser.add_argument('--youcook_train_csv', type=str, default='data/youcookii_singlef_train.csv', help='')
    parser.add_argument('--youcook_val_csv', type=str, default='data/youcookii_singlef_val.csv', help='')
    parser.add_argument('--youcook_caption_path', type=str, default='data/youcookii_caption_transcript.pickle', help='youcookii caption and transcription pickle file path')
    parser.add_argument('--youcook_features_path_2D', type=str, default='data/youcookii_videos_feature2d', help='youcookii feature path for 2D features')
    parser.add_argument('--youcook_features_path_3D', type=str, default='data/youcookii_videos_feature3d', help='youcookii feature path for 3D features')

    parser.add_argument('--pad_token', type=str, default='[PAD]', help='')

    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--visual_bert_model", default="visual-bert-base", type=str, required=False,
                        help="VisualBert pre-trained model selected in the list: visual-bert-base, visual-bert-large")
    parser.add_argument("--cross_bert_model", default="cross-bert-base", type=str, required=False,
                        help="CrossBert pre-trained model selected in the list: cross-bert-base, cross-bert-large")
    parser.add_argument("--decoder_bert_model", default="decoder-bert-base", type=str, required=False,
                        help="DecoderBert pre-trained model selected in the list: decoder-bert-base, decoder-bert-large")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.");
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `retrieval` or `caption` to finetune.")
    parser.add_argument("--datatype", default="youcook", type=str, help="Point the dataset `youcook` to finetune.")

    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=1, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")

    parser.add_argument('--cross_model', action='store_true', help="Whether training with decoder.")
    parser.add_argument('--pretrain_with_joint_sim', action='store_true', help="Whether using joint embedding when pretraining.")
    parser.add_argument('--pretrain_enhance_vmodal', action='store_true', help="Enhance visual and other modalities when pretraining.")
    parser.add_argument('--without_sim_in_decoder', action='store_true', help="Whether align in decoder when training.")
    parser.add_argument('--pretrain_without_decoder', action='store_true', help="Whether ignore decode when pretraining.")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_model", default="pytorch_model.bin.checkpoint", type=str, required=False,
                        help="Save the last model as a checkpoint.")

    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a high priority than use_mil.
        args.use_mil = True
    if args.do_pretrain is False:
        args.pretrain_without_decoder = False

    global amp_pck_loaded_
    if args.fp16:
        if not amp_pck_loaded_:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_pretrain and not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_pretrain` or `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    args.checkpoint_model = '{}_{}_{}_{}.checkpoint'.format(args.checkpoint_model, args.bert_model, args.max_words, args.max_frames)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.multiprocessing.set_sharing_strategy('file_system')   # RuntimeError: received 0 items of ancdata.

    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    args.local_rank = local_rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args, world_size, local_rank

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = VLBert.from_pretrained(args.bert_model, args.visual_bert_model, args.cross_bert_model, args.decoder_bert_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_youcook_train(args, tokenizer):
    logger.info('Loading captions: {}'.format(args.youcook_caption_path))
    caption = pickle.load(open(args.youcook_caption_path, 'rb'))
    logger.info('Done, caption length: {}'.format(len(caption)))

    youcook_dataset = Youcook_Transcript_NoPair_DataLoader(
        csv=args.youcook_train_csv,
        features_path=args.youcook_features_path_2D,
        features_path_3D=args.youcook_features_path_3D,
        caption=caption,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=-1,
        pad_token=args.pad_token,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_test(args, tokenizer):
    logger.info('Loading captions: {}'.format(args.youcook_caption_path))
    caption = pickle.load(open(args.youcook_caption_path, 'rb'))
    logger.info('Done, caption length: {}'.format(len(caption)))

    youcook_testset = Youcook_Transcript_NoPair_DataLoader(
        csv=args.youcook_val_csv,
        features_path=args.youcook_features_path_2D,
        features_path_3D=args.youcook_features_path_3D,
        caption=caption,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=-1,
        pad_token=args.pad_token,
        max_frames=args.max_frames,
    )

    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset)

def dataloader_pretrain(args, tokenizer, only_sim=False):
    logger.info('Loading captions: {}'.format(args.caption_path))
    caption = pickle.load(open(args.caption_path, 'rb'))
    logger.info('Done, caption length: {}'.format(len(caption)))

    dataset = Youtube_Transcript_DataLoader(
        csv=args.train_csv,
        features_path=args.features_path_2D,
        features_path_3D=args.features_path_3D,
        caption=caption,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=args.n_pair,
        pad_token=args.pad_token,
        max_frames=args.max_frames,
        use_mil=args.use_mil,
        only_sim=only_sim,
        sampled_use_mil=args.sampled_use_mil,
        pretrain_enhance_vmodal=args.pretrain_enhance_vmodal,
        video_dim=args.video_dim,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return dataloader, len(dataset), sampler

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, local_rank, type_name="", global_step=-1, optimizer=None):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)

    if global_step != -1 and optimizer is not None:
        amp_state_dict = {}
        if args.fp16:
            amp_state_dict = amp.state_dict()
        state_dict = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'last_optimizer_state': convert_state_dict_type(optimizer.state_dict()),
            'amp_state_dict': amp_state_dict,
        }
        checkpoint_model_file = os.path.join(args.output_dir, args.checkpoint_model)
        torch.save(state_dict, checkpoint_model_file)
        logger.info("Checkpoint is saved. use `load_checkpoint` to recovery it.")

    return output_model_file

def load_model(epoch, args, n_gpu, device, model, global_step=0, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))

    last_optim_state = None
    amp_state_dict = None
    checkpoint_model_file = os.path.join(args.output_dir, args.checkpoint_model)
    if epoch == -1 and args.load_checkpoint and os.path.exists(checkpoint_model_file):
        checkpoint_state = torch.load(checkpoint_model_file, map_location='cpu')
        epoch = checkpoint_state['epoch']
        global_step = checkpoint_state['global_step']
        model_state_dict = checkpoint_state['model_state_dict']
        last_optim_state = checkpoint_state['last_optimizer_state']
        if args.fp16 and 'amp_state_dict' in checkpoint_state:
            amp_state_dict = checkpoint_state['amp_state_dict']
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = VLBert.from_pretrained(args.bert_model, args.visual_bert_model, args.cross_bert_model, args.decoder_bert_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
        logger.info("Checkpoint loaded from %s", checkpoint_model_file)
    elif os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = VLBert.from_pretrained(args.bert_model, args.visual_bert_model, args.cross_bert_model, args.decoder_bert_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)

    return epoch, global_step, last_optim_state, amp_state_dict, model

def train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        # if n_gpu == 1:
        #     # multi-gpu does scattering it-self
        #     batch = tuple(t.to(device) for t in batch)
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        loss = model(input_ids, segment_ids, input_mask, video, video_mask,
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                     output_caption_ids=pairs_output_caption_ids)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss) * args.gradient_accumulation_steps,
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return (active_sequence_output_rpt, active_visual_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
        sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        dec_output = decoder(sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt,
                             video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, rougeObj=None, nlgEvalObj=None, test_set=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._choice_sim:
        return 0.

    all_result_lists = []
    all_caption_lists = []
    model.eval()
    for batch in test_dataloader:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        with torch.no_grad():
            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)
            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = sequence_output.device
            n_inst, len_s, d_h = sequence_output.size()
            _, len_v, v_h = visual_output.size()

            decoder = model.decoder_caption         # This is a decoder function

            # Note: shaped first, then decoder need the parameter shaped=True
            input_ids = input_ids.view(-1, input_ids.shape[-1])  # [batch_size*n_pair, self.max_words]
            input_mask = input_mask.view(-1, input_mask.shape[-1])  # [batch_size*n_pair, self.max_words]
            video_mask = video_mask.view(-1, video_mask.shape[-1])  # [batch_size*n_pair, self.max_frames]

            sequence_output_rpt = sequence_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            input_ids_rpt = input_ids.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            input_mask_rpt = input_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1):
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt))

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt), \
                inst_idx_to_position_map = collate_active_info((sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                all_caption_lists.append(decode_text)

    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.caption[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    # Filter out hyps of 0 length
    hyps_and_refs = zip(all_result_lists, all_caption_lists)
    hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0
                     and len([" ".join(sub_s.split()) for sub_s in _[0].split(".") if len(sub_s) > 0]) > 0]
    all_result_lists, all_caption_lists = zip(*hyps_and_refs)

    # Evaluate
    metrics_dict = rougeObj.get_scores(hyps=all_result_lists, refs=all_caption_lists, avg=True, ignore_empty=True)
    logger.info(">>>  rouge_1f: {:.4f}, rouge_2f: {:.4f}, rouge_lf: {:.4f}".
                format(metrics_dict["rouge-1"]["f"], metrics_dict["rouge-2"]["f"], metrics_dict["rouge-l"]["f"]))

    metrics_nlg = nlgEvalObj.compute_metrics(ref_list=[all_caption_lists], hyp_list=all_result_lists)
    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))

    Bleu_4 = metrics_nlg["Bleu_4"]
    return Bleu_4

DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_train, "val":dataloader_youcook_test}

def main():
    global logger
    args = get_args()
    args, world_size, local_rank = set_seed_logger(args)
    device, n_gpu = init_device(args, local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.do_train:
        args.task_type = "caption"
    model = init_model(args, device, n_gpu, local_rank)
    only_sim = model.module._choice_sim if hasattr(model, 'module') else model._choice_sim

    if args.task_type == "caption":
        rougeObj = Rouge()
        nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

    datatype = "youcook"
    assert datatype in DATALOADER_DICT
    if args.do_pretrain is False:
        test_dataloader, test_length = DATALOADER_DICT[datatype]["val"](args, tokenizer)
        if local_rank == 0:
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_pretrain:
        train_dataloader, train_length, sampler = dataloader_pretrain(args, tokenizer, only_sim=only_sim)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        global_step = 0
        epoch = -1
        last_optim_state = None
        amp_state_dict = None
        if args.load_checkpoint:
            epoch, global_step, last_optim_state, amp_state_dict, model = load_model(epoch, args, n_gpu, device, model, global_step=global_step)
            epoch += 1
            if local_rank == 0:
                logger.warning("Will continue to epoch: {}".format(epoch))
        epoch = 0 if epoch < 0 else epoch

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0

        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=coef_lr)
        if last_optim_state is not None:
            optimizer.load_state_dict(last_optim_state)
        if amp_state_dict is not None:
            amp.load_state_dict(amp_state_dict)

        if local_rank == 0:
            logger.info("***** Running pretraining *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        iter_ls_ = [itm for itm in range(args.epochs) if itm >= epoch]
        for epoch in iter_ls_:
            sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=local_rank)

            if local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                save_model(epoch, args, model, local_rank, type_name="pretrain", global_step=global_step, optimizer=optimizer)
    elif args.do_train:

        train_dataloader, train_length, train_sampler = DATALOADER_DICT[datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=coef_lr)

        if local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=local_rank)

            if local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, local_rank, type_name="")
                if epoch > 0:
                    Bleu_4 = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, rougeObj=rougeObj, nlgEvalObj=nlgEvalObj)
                    if best_score <= Bleu_4:
                        best_score = Bleu_4
                        best_output_model_file = output_model_file
                    logger.info("The best model is: {}, the Bleu_4 is: {:.4f}".format(best_output_model_file, best_score))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if local_rank == 0:
            _, _, _, _, model = load_model(-1, args, n_gpu, device, model, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, rougeObj=rougeObj, nlgEvalObj=nlgEvalObj)
    elif args.do_eval:
        if local_rank == 0:
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, rougeObj=rougeObj, nlgEvalObj=nlgEvalObj)

if __name__ == "__main__":
    main()