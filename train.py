import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv
import ast
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
from segment_anything import sam_model_registry
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from multiprocessing import Manager
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import logging
from data_loader import get_multi_loader
from checkpoint_utils import upgrade_legacy_state_dict
from model import SegMoTE
from utils import FocalDice_MSELoss
from torch.nn import CrossEntropyLoss
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='segmote_train')

parser.add_argument("--data_dir", type=str, default='dataset')
parser.add_argument("--dataset_list", type=str, nargs='+', default=['BTCV'])
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--test_mode', action='store_true')
parser.add_argument('--batch_size', type=int, default=10)

parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--sam_checkpoint', type=str, default=None)
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--text_tokenizer_name', type=str, default='openai/clip-vit-base-patch32')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mask_num', type=int, default=2)
parser.add_argument('--inter_num', type=int, default=4)

parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--lr_scheduler', type=str, default=None)
parser.add_argument('--step_size', type=list, default=[7,12])
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--port', type=int, default=12307)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--dist', action='store_true', help='distributed training or not')
parser.add_argument('-num_workers', type=int, default=1)
parser.add_argument('--group_a', type=str, nargs='*', default=['isic2016_task1','isic2017_task1','isic2018_task1'],
                    help='datasets that can mix within a batch but cannot mix with others')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)

device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def build_model(args):
    category_weights = 'dataloaders/categories_weight.pkl'
    sam = sam_model_registry[args.model_type](args).to(args.device)
    segmote = SegMoTE(
        sam,
        test_mode=args.test_mode,
        select_mask_num=args.mask_num,
        category_weights=category_weights,
        text_tokenizer_name=args.text_tokenizer_name,
    ).to(args.device)

    if args.multi_gpu:
        segmote = DDP(segmote, device_ids=[args.rank], output_device=args.rank)
    return segmote


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path, args.resume)
        else:
            self.start_epoch = 0

    def set_loss_fn(self):
        self.seg_loss = FocalDice_MSELoss()
        self.ce_loss = CrossEntropyLoss()


    def set_optimizer(self):
        params = (p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def load_checkpoint(self, ckp_path, resume):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)

        if last_ckpt:
            state_dict = upgrade_legacy_state_dict(last_ckpt['model_state_dict'])
            target_model = self.model.module if self.args.multi_gpu else self.model
            target_model.load_state_dict(state_dict, strict=False)
            if resume:
                self.start_epoch = last_ckpt.get('epoch', 0)
                if 'optimizer_state_dict' in last_ckpt:
                    self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                if 'lr_scheduler_state_dict' in last_ckpt:
                    self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt.get('losses', [])
                self.dices = last_ckpt.get('dices', [])
                self.ious = last_ckpt.get('ious', [])
                self.best_loss = last_ckpt.get('best_loss', np.inf)
                self.best_dice = last_ckpt.get('best_dice', 0.0)
            else:
                self.start_epoch = 0
            print(f"Loaded checkpoint from {ckp_path}")

        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")


    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "ious": self.ious,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "best_dice": self.best_dice,
            "args": self.args,
        }, join(MODEL_SAVE_PATH, f"segmote_{describe}.pth"))


    def get_iou_and_dice(self, pred, label):
        assert pred.shape == label.shape
        pred = (torch.sigmoid(pred) > 0.5)
        label = (label > 0)
        intersection = torch.logical_and(pred, label).sum(dim=(1, 2, 3))
        union = torch.logical_or(pred, label).sum(dim=(1, 2, 3))
        iou = intersection.float() / (union.float() + 1e-8)
        dice = (2 * intersection.float()) / (pred.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) + 1e-8)
        return iou.mean().item(), dice.mean().item()

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def interaction(
        self,
        model,
        image_embedding,
        gt_low_masks,
        pseudo_low_masks,
        gt_preds,
        pseudo_preds,
        labels,
        pseudos,
        interm_embeddings,
        if_prompt,
                ):

        total_loss = 0
        text_and_mask_inter = np.random.randint(0, self.args.inter_num-1)
        with amp.autocast():
            for inter in range(self.args.inter_num):
                if inter == text_and_mask_inter or inter == self.args.inter_num-1:
                    gt_prompts = model.process_mask_prompt(gt_low_masks)
                    gt_prompts.update(self.text_prompt)

                    gt_outputs,loss_moe,_ = model.forward_decoder(image_embedding, gt_prompts,interm_embeddings=interm_embeddings,if_prompt=if_prompt)
                    gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
                    gt_loss = self.seg_loss(gt_preds, labels.float(), gt_outputs['iou_pred'])

                    pseudo_prompts = model.process_mask_prompt(pseudo_low_masks)

                else:
                    gt_prompts = model.supervised_prompts(None, labels, gt_preds, gt_low_masks, 'points')
                    if random.random() > 0.6:
                        gt_prompts.update(self.text_prompt)
                        del gt_prompts['mask_inputs']

                    gt_outputs,loss_moe,_ = model.forward_decoder(image_embedding, gt_prompts,interm_embeddings=interm_embeddings,if_prompt=if_prompt)
                    gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
                    gt_loss = self.seg_loss(gt_preds, labels.float(), gt_outputs['iou_pred'])

                    pseudo_prompts = model.unsupervised_prompts(pseudos, pseudo_preds, pseudo_low_masks, 'points')

                pseudo_outputs,loss_ps,_ = model.forward_decoder(image_embedding,  pseudo_prompts,interm_embeddings=interm_embeddings,inter=True)
                pseudo_preds, pseudo_low_masks = pseudo_outputs['masks'], pseudo_outputs['low_res_masks']
                pseudo_loss = self.seg_loss(pseudo_preds, pseudos.float(), pseudo_outputs['iou_pred'])

                loss = gt_loss + pseudo_loss+0.01*loss_moe
                if torch.isnan(loss).any():
                    print(f"Detected NaN loss. Skipping this inter.")
                    total_loss += 0
                    continue
                else:
                    total_loss += loss.item()
                    self.scaler.scale(loss).backward(retain_graph=True)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        loss = total_loss / self.args.inter_num
        return loss, gt_preds, pseudo_preds


    def train_epoch(self, epoch):
        step_loss, step_iou, step_dice = 0, 0, 0
        self.model.train()

        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model

        tbar = tqdm(self.dataloaders)
        l = len(self.dataloaders)


        for step, batch_input in enumerate(tbar):
            images, labels = batch_input["image"].to(device), batch_input["label"].to(device).type(torch.long)
            pseudos = batch_input["pseudo"].to(device)
            self.target_list = batch_input['target_list']
            npromt = batch_input['npromt']
            gt_prompt = batch_input["gt_prompt"]
            pseudo_prompt = batch_input["pseudo_prompt"]
            if_prompt = batch_input['is_group_a']
            dataset_codes=batch_input['dataset_codes']

            gt_prompts, pseudo_prompts = {}, {}
            if torch.sum(labels) == 0 or torch.sum(pseudos) == 0:
                continue

            self.text_prompt = model.process_text_prompt(self.target_list)

            self.img_shape = images.shape
            image_embedding,interm_embeddings = model.image_forward(images)

            gt_prm = random.choices(['bboxes', 'points', 'text'], [0.4, 0.3, 0.3])[0]
            pse_prm = random.choices(['bboxes', 'points'], [0.5, 0.5])[0]

            if gt_prm == 'bboxes':
                gt_prompts['bboxes'] = gt_prompt['bboxes'].to(device)
            elif gt_prm == 'points':
                gt_prompts['point_coords'] = gt_prompt['point_coords'].to(device)
                gt_prompts['point_labels'] = gt_prompt['point_labels'].to(device)
            else:
                gt_prompts.update(self.text_prompt)

            if pse_prm == 'bboxes':
                pseudo_prompts['bboxes'] = pseudo_prompt['bboxes'].to(device)
            else:
                pseudo_prompts['point_coords'] = pseudo_prompt['point_coords'].to(device)
                pseudo_prompts['point_labels'] = pseudo_prompt['point_labels'].to(device)

            with amp.autocast():
                gt_outputs,loss_moe,moe_idx = model.forward_decoder(image_embedding, gt_prompts,interm_embeddings=interm_embeddings)
                gt_loss = self.seg_loss(gt_outputs['masks'], labels.float(), gt_outputs['iou_pred'])

                pseudo_outputs,loss_moe_ps,_ = model.forward_decoder(image_embedding, pseudo_prompts,interm_embeddings=interm_embeddings)
                pseudo_loss = self.seg_loss(pseudo_outputs['masks'], pseudos.float(), pseudo_outputs['iou_pred'])

                loss = gt_loss + pseudo_loss+0.01*loss_moe


                if torch.isnan(loss).any():
                    print(f"Detected NaN loss at epoch {epoch}, batch {step}. Skipping this batch.")
                    continue
                else:
                    self.scaler.scale(loss).backward(retain_graph=False)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()


            gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
            pseudo_preds, pseudo_low_masks = pseudo_outputs['masks'], pseudo_outputs['low_res_masks']

            image_embedding = image_embedding.detach().clone()
            self.text_prompt['text_inputs'] = self.text_prompt['text_inputs'].detach().clone()

            loss, gt_preds, pseudo_preds = self.interaction(model, image_embedding, gt_low_masks, pseudo_low_masks,
                                                            gt_preds, pseudo_preds,
                                                            labels, pseudos,interm_embeddings,if_prompt
                                                            )

            gt_iou, gt_dice = self.get_iou_and_dice(gt_preds, labels)


            step_loss += loss
            step_iou += gt_iou
            step_dice += gt_dice

        if self.args.multi_gpu:
            dist.barrier()
            local_loss = torch.tensor([step_loss / l]).to(self.args.device)
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            avg_loss = local_loss.item() / dist.get_world_size()

            local_iou = torch.tensor([float(step_iou / l)]).to(self.args.device)
            dist.all_reduce(local_iou, op=dist.ReduceOp.SUM)
            avg_iou = local_iou.item() / dist.get_world_size()

            local_dice = torch.tensor([float(step_dice / l)]).to(self.args.device)
            dist.all_reduce(local_dice, op=dist.ReduceOp.SUM)
            avg_dice = local_dice.item() / dist.get_world_size()
        else:
            avg_loss, avg_iou, avg_dice = step_loss / l, step_iou / l, step_dice / l

        return avg_loss, avg_iou, avg_dice

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)

            avg_loss, avg_iou, avg_dice = self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(avg_loss)
                self.ious.append(avg_iou)
                self.dices.append(avg_dice)
                print(f'Epochs: {epoch}, LR: {self.lr_scheduler.get_last_lr()}, Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}')
                logger.info(f'Epoch\t {epoch}\t LR\t {self.lr_scheduler.get_last_lr()}\t: loss: {avg_loss:.4f}, iou: {avg_iou:.4f}, dice: {avg_dice:.4f}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                self.save_checkpoint(epoch, state_dict, describe='latest')

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss


                if avg_iou > self.best_iou:
                    self.best_iou = avg_iou

                self.save_checkpoint(epoch, state_dict, describe=f'epoch_{epoch}')

                if avg_dice > self.best_dice:
                    self.best_dice = avg_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                self.plot_result(self.losses, 'Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                self.plot_result(self.ious, 'IoU', 'IoU')

        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}, Best iou: {self.best_iou}, Best dice: {self.best_dice}')
        logger.info(f'args : {self.args}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

def device_config(args):
    try:
        if not args.multi_gpu:
            if args.device == 'cuda':
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
            else:
                args.device = torch.device(args.device)
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node
    except RuntimeError as e:
        print(e)

def main():
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        dataloaders = get_multi_loader(args)

        model = build_model(args)

        trainer = BaseTrainer(model, dataloaders, args)

        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2024 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_multi_loader(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    os.environ['NCCL_TIMEOUT_MS'] = '3600000'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size,timeout=datetime.timedelta(seconds=36000000))

def cleanup():
    dist.destroy_process_group()


import csv, os
from collections import defaultdict
def _bytes_per_dtype(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4, torch.float: 4,
        torch.float16: 2, torch.bfloat16: 2,
        torch.int64: 8, torch.long: 8,
        torch.int32: 4, torch.int16: 2,
        torch.uint8: 1, torch.bool: 1
    }.get(dtype, 4)

def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def human_m(n: int) -> str:
    return f"{n:,}  (~{n/1e6:.2f} M)"

def print_model_stats(model: torch.nn.Module,
                      args,
                      save_bucket_csv: str = None,
                      detail_prefix: str = None,
                      save_detail_csv: str = None):


    if getattr(args, "multi_gpu", False) and getattr(args, "rank", 0) != 0:
        return

    m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


    total, trainable = count_params(m)
    print("\n========== Model Parameter Stats ==========")
    print(f"Total params:     {human_m(total)}")
    print(f"Trainable params: {human_m(trainable)}")
    print(f"Non-trainable:    {human_m(total - trainable)}")
    print("===========================================\n")


    bucket = defaultdict(int)
    for name, p in m.named_parameters():
        if p.requires_grad:
            top = name.split('.')[0] if '.' in name else name
            bucket[top] += p.numel()

    print("---- Trainable params by top-level module ----")
    for k, v in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:20s}: {v/1e6:6.2f} M  ({v:,})")
    print("----------------------------------------------")


    if save_bucket_csv:
        os.makedirs(os.path.dirname(save_bucket_csv), exist_ok=True)
        with open(save_bucket_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["module", "num_params", "num_params_M"])
            for k, v in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True):
                w.writerow([k, v, round(v/1e6, 6)])
        print(f"[Info] Exported parameter summary to: {save_bucket_csv}")


    if detail_prefix is not None:
        rows = []
        total_pref = 0
        train_pref = 0
        for name, p in m.named_parameters():
            if not name.startswith(detail_prefix + ".") and name != detail_prefix:
                continue
            n = p.numel()
            total_pref += n
            if p.requires_grad:
                train_pref += n
            nb = n * _bytes_per_dtype(p.dtype)
            rows.append((name, tuple(p.shape), str(p.dtype), p.requires_grad, n, nb))

        rows.sort(key=lambda x: x[4], reverse=True)

        print(f"\n==== '{detail_prefix}' parameter details (top 30 by size) ====")
        for name, shape, dtype, req, n, nb in rows[:30]:
            print(f"{name:70s} shape={str(shape):18s} dtype={dtype:12s} "
                  f"trainable={str(req):5s} params={n:,} (~{n/1e6:.2f} M), ~{nb/1e6:.2f} MB")
        print("--------------------------------------------------")
        print(f"{detail_prefix} total:     {human_m(total_pref)}")
        print(f"{detail_prefix} trainable: {human_m(train_pref)}")
        print(f"{detail_prefix} frozen:    {human_m(total_pref - train_pref)}")

        if save_detail_csv:
            os.makedirs(os.path.dirname(save_detail_csv), exist_ok=True)
            with open(save_detail_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name","shape","dtype","trainable","num_params","num_params_M","approx_MB"])
                for name, shape, dtype, req, n, nb in rows:
                    w.writerow([name, str(shape), dtype, req, n, round(n/1e6,6), round(nb/1e6,6)])
            print(f"[Info] Exported {detail_prefix} details to: {save_detail_csv}")


def freeze_decoder_except_new(segmote, trainable_keywords=None):


    default_keywords = [

        'mlp_moe_token',
        'experts', 'experts_token',
        'proj_down',
        'dte', 'dteall',
        'proj$', 'proj_token$',
        'dte_token', 'dteall_token',
        'expert_tokens', 'compress_vit_feat', 'expert_embedding_encoder', 'expert_feature_fusion',


        'expert_mlp',
        'img_prompt_pool'


    ]

    import re
    keywords = default_keywords if trainable_keywords is None else trainable_keywords

    m = segmote.module if isinstance(segmote, torch.nn.parallel.DistributedDataParallel) else segmote
    dec = m.mask_decoder


    for _, p in dec.named_parameters():
        p.requires_grad = False


    def _match(name: str) -> bool:
        for kw in keywords:
            if kw.endswith('$'):
                if re.search(kw, name):
                    return True
            elif kw in name:
                return True
        return False

    kept = []
    for n, p in dec.named_parameters():
        if _match(n):
            p.requires_grad = True
            kept.append(n)


    trainable_cnt = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    total_cnt = sum(p.numel() for p in dec.parameters())
    print(f"[freeze] mask_decoder trainable: {trainable_cnt:,} / {total_cnt:,} params")


if __name__ == '__main__':
    main()
