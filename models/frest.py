import copy
import itertools
import math
import os
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from PIL import Image
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.cli import instantiate_class

from .backbones.mix_transformer import DropPath
from .heads.base import BaseHead
from .utils import (colorize_mask, estimate_probability_of_confidence_interval,
                    warp)
from .backbones.condition_projector import ConditionProjector 
from torch.backends import cudnn


class FREST(pl.LightningModule):
    """Feature RESToration for source-free adaptation to adverse conditions.

    Each iteration alternates two steps (Sec. 4):
      Step 1 - learn the condition embedding space: train the condition strainer
        (psi_strainer) and projection head (psi_proj) with the condition-specific
        loss L_spec (Eq. 1-2) and the self-training loss L_self.
      Step 2 - feature restoration: train the segmentation network with the
        feature restoration loss L_resto (Eq. 3), the adverse condition
        discriminating loss L_dis (Eq. 4), L_self and the entropy loss L_ent.
    Only the encoder (phi_enc) and decoder (phi_dec) are used at inference; the
    strainer is skipped so the model runs on the restored features f_adv.
    """

    def __init__(self,
                 optimizer_init: dict,
                 lr_scheduler_init: dict,
                 backbone: nn.Module,
                 head: BaseHead,
                 contrastive_head: Optional[nn.Module] = None,
                 alignment_backbone: Optional[nn.Module] = None,
                 alignment_head: Optional[BaseHead] = None,
                 self_training_loss: Optional[nn.Module] = None,
                 self_training_loss_weight: float = 1.0,
                 entropy_loss: Optional[nn.Module] = None,
                 entropy_loss_weight: float = 1.0,
                 contrastive_loss: Optional[nn.Module] = None,
                 contrastive_loss_out: Optional[nn.Module] = None,
                 contrastive_loss_weight: float = 1.0,
                 use_slide_inference: bool = True,
                 use_ensemble_inference: bool = True,
                 epoch_search: int = 1,
                 discriminator_weight: float = 0.1,
                 inference_ensemble_scale: List[float] = [0.8, 1.0, 1.2],
                 inference_ensemble_scale_divisor: Optional[int] = None,
                 projection_head_lr_factor: float = 10.0,
                 head_lr_factor: float = 1.0,
                 ema_momentum: float = 0.9999,
                 projector_lr: float = 0.001,
                 grad_clip: float = 0.0,
                 use_seg: bool = True,
                 freeze_decoder: bool = True,
                 metrics: dict = {},
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'backbone',
            'head',
            'contrastive_head',
            'alignment_backbone',
            'alignment_head',
            'self_training_loss',
            'entropy_loss',
            'contrastive_loss',
            'contrastive_loss_out'
        ])

        cudnn.benchmark = True
        cudnn.deterministic = False
        #### MODEL ####
        self.backbone = backbone
        self.head = head
        self.contrastive_head = contrastive_head
        self.m_backbone = copy.deepcopy(self.backbone)
        self.m_contrastive_head = copy.deepcopy(self.contrastive_head)
        for ema_m in filter(None, [self.m_backbone, self.m_contrastive_head]):
            for param in ema_m.parameters():
                param.requires_grad = False
        self.alignment_backbone = alignment_backbone
        self.alignment_head = alignment_head
        for alignment_m in filter(None, [self.alignment_backbone, self.alignment_head]):
            for param in alignment_m.parameters():
                param.requires_grad = False
        self.head_lr_factor = head_lr_factor

        #### LOSSES ####
        self.self_training_loss = self_training_loss
        self.self_training_loss_weight = self_training_loss_weight
        self.entropy_loss = entropy_loss
        self.entropy_loss_weight = entropy_loss_weight
        self.contrastive_loss = contrastive_loss
        self.contrastive_loss_out = contrastive_loss_out
        self.contrastive_loss_weight = contrastive_loss_weight
        self.discriminator_weight = discriminator_weight

        self.grad_clip = grad_clip
        self.use_seg = use_seg
        self.automatic_optimization = False

        #### INFERENCE ####
        self.use_slide_inference = use_slide_inference
        self.use_ensemble_inference = use_ensemble_inference
        self.inference_ensemble_scale = inference_ensemble_scale
        self.inference_ensemble_scale_divisor = inference_ensemble_scale_divisor

        #### METRICS ####
        val_metrics = {'val_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('val', {}).items() for el in m}
        test_metrics = {'test_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('test', {}).items() for el in m}
        self.valid_metrics = torchmetrics.MetricCollection(val_metrics)
        self.test_metrics = torchmetrics.MetricCollection(test_metrics)

        #### OPTIMIZATION ####
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        self.epoch_search = epoch_search
        #### OTHER STUFF ####
        self.projection_head_lr_factor = projection_head_lr_factor
        self.ema_momentum = ema_momentum

        
        self.projector_lr = projector_lr
        self.condition_projector = ConditionProjector(128)
        self.condition_projector.cuda()


    def training_step(self, batch, batch_idx):


        optimizer_seg, optimizer_strainer, optimizer_projector = self.optimizers()
        lr_scheduler_seg, lr_scheduler_strainer = self.lr_schedulers()

        #
        # LOSSES
        #
        loss = torch.tensor(0.0, device=self.device)
        self_training_loss_val = torch.tensor(0.0, device=self.device)
        entropy_loss_val = torch.tensor(0.0, device=self.device)
        restoration_loss = torch.tensor(0.0, device=self.device)
        condition_specific_loss = torch.tensor(0.0, device=self.device)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.condition_projector.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = False
        
        optimizer_projector.zero_grad()

        # import pdb; pdb.set_trace()
        images_trg = batch[0]['image']
        pseudo_label_trg = batch[0]['semantic']
        images_ref = batch[0]['image_ref']

        feats_trg = self.backbone(images_trg, img='trg')[0]
        logits_trg = self.head(feats_trg)
        logits_trg = nn.functional.interpolate(
            logits_trg, images_trg.shape[-2:], mode='bilinear', align_corners=False)

        if self.contrastive_loss_out is not None:
            contrastive_inp = feats_trg
            emb_anc = self.contrastive_head(contrastive_inp)
            with torch.no_grad():
                m_input = torch.cat((images_trg, images_ref))
                m_output = self.m_contrastive_head(self.m_backbone(m_input)[0])
                emb_neg, emb_pos = torch.tensor_split(m_output, 2)
                emb_pos, confidence = self.align(emb_pos, images_ref, images_trg)


        # CONTRASTIVE

        
        emb_anc_out, emb_pos_out, emb_negs_out = self.contrastive_loss_out(emb_anc, emb_pos, confidence)
        if emb_anc_out.shape[0] != 0:
            emb_anc_cond = self.condition_projector(emb_anc_out)
            emb_pos_cond = self.condition_projector(emb_pos_out)
            emb_negs_out = torch.transpose(emb_negs_out, 0, 1)
            emb_negs_cond = self.condition_projector(emb_negs_out)
            emb_negs_cond = torch.transpose(emb_negs_cond, 0, 1)



            l_pos_denses = torch.einsum('nc,ck->nk', [emb_anc_cond, emb_negs_cond])
            l_neg_dense = torch.einsum('nc,nc->n', [emb_anc_cond, emb_pos_cond]).unsqueeze(-1)


            max_sim_indices = torch.argmax(l_pos_denses, dim=1, keepdim=True)
            max_pos_denses = l_pos_denses.gather(1, max_sim_indices)
            logits = torch.cat((l_neg_dense, max_pos_denses), dim=1) / 0.7

            labels = torch.ones(logits.size(0), dtype=torch.long, device=logits.device) 
            condition_specific_loss += self.contrastive_loss_weight * nn.functional.cross_entropy(logits, labels, reduction='mean')


            if condition_specific_loss != 0:
                self.manual_backward(condition_specific_loss)
                optimizer_projector.step()


            # update the queue
        condition_specific_loss = torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        self_training_loss_val = torch.tensor(0.0, device=self.device)
        entropy_loss_val = torch.tensor(0.0, device=self.device)

        optimizer_strainer.zero_grad()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if 'strainer' in name:
                param.requires_grad = True
        for param in self.condition_projector.parameters():
            param.requires_grad = False     
        for param in self.head.parameters():
            param.requires_grad = False

        feats_trg = self.backbone(images_trg, img='trg')[0]
        logits_trg = self.head(feats_trg)
        logits_trg = nn.functional.interpolate(
            logits_trg, images_trg.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.use_seg == True:
            if self.self_training_loss is not None:
                self_training_loss = self.self_training_loss(logits_trg, pseudo_label_trg)
                self_training_loss *= self.self_training_loss_weight
                self.log("train_self_training_loss_valelftraining", self_training_loss)
                self_training_loss_val += self_training_loss

        if self.contrastive_loss_out is not None:
            contrastive_inp = feats_trg
            emb_anc = self.contrastive_head(contrastive_inp)
            with torch.no_grad():
                m_input = torch.cat((images_trg, images_ref))
                m_output = self.m_contrastive_head(self.m_backbone(m_input)[0])
                emb_neg, emb_pos = torch.tensor_split(m_output, 2)
                emb_pos, confidence = self.align(emb_pos, images_ref, images_trg)


        emb_anc_out, emb_pos_out, emb_negs_out = self.contrastive_loss_out(emb_anc, emb_pos, confidence)
        if emb_anc_out.shape[0] != 0:
            emb_anc_cond = self.condition_projector(emb_anc_out)
            emb_pos_cond = self.condition_projector(emb_pos_out)
            emb_negs_out = torch.transpose(emb_negs_out, 0, 1)
            emb_negs_cond = self.condition_projector(emb_negs_out)
            emb_negs_cond = torch.transpose(emb_negs_cond, 0, 1)


            l_pos_denses = torch.einsum('nc,ck->nk', [emb_anc_cond, emb_negs_cond])
            l_neg_dense = torch.einsum('nc,nc->n', [emb_anc_cond, emb_pos_cond]).unsqueeze(-1)


            max_sim_indices = torch.argmax(l_pos_denses, dim=1, keepdim=True)
            max_pos_denses = l_pos_denses.gather(1, max_sim_indices)
            logits = torch.cat((l_neg_dense, max_pos_denses), dim=1) / 0.7

            labels = torch.ones(logits.size(0), dtype=torch.long, device=logits.device) 
            condition_specific_loss += self.contrastive_loss_weight * nn.functional.cross_entropy(logits, labels, reduction='mean')

            


            with torch.no_grad():
                emb_neg = self.all_gather(emb_neg)
                if emb_neg.dim() == 5:
                    emb_neg = torch.flatten(emb_neg, start_dim=0, end_dim=1)
                self.contrastive_loss_out.update_queue(emb_neg)

        if self.use_seg == False:
            self_training_loss_val = 0
        
        loss += condition_specific_loss + self_training_loss_val
        if loss != 0:
            self.manual_backward(loss)
            optimizer_strainer.step()
            lr_scheduler_strainer.step()

        if self.current_epoch > self.epoch_search:


            condition_specific_loss = torch.tensor(0.0, device=self.device)
            loss = torch.tensor(0.0, device=self.device)
            self_training_loss_val = torch.tensor(0.0, device=self.device)
            entropy_loss_val = torch.tensor(0.0, device=self.device)
            discriminating_loss = torch.tensor(0.0, device=self.device)
            
            optimizer_seg.zero_grad()

            for param in self.backbone.parameters():
                param.requires_grad = True
            for name, param in self.backbone.named_parameters():
                if 'strainer' in name:
                    param.requires_grad = False
            for param in self.condition_projector.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = True

            feats_trg, disc_loss = self.backbone(images_trg, img='trg', discriminate=True)
            logits_trg = self.head(feats_trg)
            logits_trg = nn.functional.interpolate(
                logits_trg, images_trg.shape[-2:], mode='bilinear', align_corners=False)
    

            if self.contrastive_loss_out is not None:
                if self.global_step < self.contrastive_loss_out.warm_up_steps + 4000:
                    if isinstance(feats_trg, (list, tuple)):
                        contrastive_inp = [el.detach() for el in feats_trg]
                    else:
                        contrastive_inp = feats_trg.detach()
                else:
                    contrastive_inp = feats_trg
                emb_anc = self.contrastive_head(contrastive_inp)
                with torch.no_grad():
                    self.update_momentum_encoder()
                    m_input = torch.cat((images_trg, images_ref))
                    m_output = self.m_contrastive_head(self.m_backbone(m_input)[0])
                    emb_neg, emb_pos = torch.tensor_split(m_output, 2)
                    # warp the reference embeddings
                    emb_pos, confidence = self.align(emb_pos, images_ref, images_trg)

            # SELF-TRAINING / CONSISTENCY
            if self.self_training_loss is not None:
                self_training_loss = self.self_training_loss(logits_trg, pseudo_label_trg)
                self_training_loss *= self.self_training_loss_weight
                self.log("train_self_training_loss_valelftraining", self_training_loss)
                self_training_loss_val += self_training_loss

            # ENTROPY MINIMIZATION
            if self.entropy_loss is not None:
                entropy_loss = self.entropy_loss(logits_trg)
                entropy_loss *= self.entropy_loss_weight
                self.log("train_entropy_loss_valntropy", entropy_loss)
                entropy_loss_val += entropy_loss

            
            discriminating_loss += self.discriminator_weight * disc_loss

            

            emb_anc_c, emb_pos_c, emb_negs_c = self.contrastive_loss_out(emb_anc, emb_pos, confidence)
            emb_anc_cond_c = self.condition_projector(emb_anc_c)
            emb_pos_cond_c = self.condition_projector(emb_pos_c)
            emb_negs_c = torch.transpose(emb_negs_c, 0, 1)
            emb_negs_cond_c = self.condition_projector(emb_negs_c)
            emb_negs_cond_c = torch.transpose(emb_negs_cond_c, 0, 1)
            
            restoration_loss += torch.nn.L1Loss(reduction='mean')(emb_anc_cond_c, emb_pos_cond_c)     
            


            loss += self_training_loss_val + entropy_loss_val + restoration_loss + discriminating_loss

            
            self.manual_backward(loss)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 2000)
            optimizer_seg.step()
            lr_scheduler_seg.step() 


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['image'], batch['semantic']
        y_hat = self.forward(x, out_size=y.shape[-2:])
        src_name = self.trainer.datamodule.idx_to_name['val'][dataloader_idx]
        for k, m in self.valid_metrics.items():
            if src_name in k:
                m.update(y_hat, y)  

    def validation_epoch_end(self, outs):
        out_dict = self.valid_metrics.compute()
        for k, v in out_dict.items():
            self.log(k, v)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['image'], batch['semantic']
        y_hat = self.forward(x, out_size=y.shape[-2:])
        src_name = self.trainer.datamodule.idx_to_name['test'][dataloader_idx]
        for k, m in self.test_metrics.items():
            if src_name in k:
                m.update(y_hat, y)

    def test_epoch_end(self, outs):
        out_dict = self.test_metrics.compute()
        for k, v in out_dict.items():
            self.log(k, v)
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.trainer.datamodule.predict_on[dataloader_idx]
        save_dir = os.path.join(os.path.dirname(
            self.ckpt_dir), 'preds', dataset_name)
        col_save_dir = os.path.join(os.path.dirname(
            self.ckpt_dir), 'color_preds', dataset_name)
        if self.trainer.is_global_zero:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(col_save_dir, exist_ok=True)
        img_names = batch['filename']
        x = batch['image'] if 'image' in batch.keys() else batch['image_ref']
        orig_size = self.trainer.datamodule.predict_ds[dataloader_idx].orig_dims
        y_hat = self.forward(x, orig_size)
        preds = torch.argmax(y_hat, dim=1)
        for pred, im_name in zip(preds, img_names):
            arr = pred.cpu().numpy()
            image = Image.fromarray(arr.astype(np.uint8))
            image.save(os.path.join(
                save_dir, '.'.join([*im_name.split('.')[:-1], 'png'])))
            col_image = colorize_mask(image)
            col_image.save(os.path.join(
                col_save_dir, '.'.join([*im_name.split('.')[:-1], 'png'])))

    def forward(self, x, out_size=None):
        if self.use_ensemble_inference:
            out = self.forward_ensemble(x, out_size)
        else:
            if self.use_slide_inference:
                out = self.slide_inference(x)
            else:
                out = self.whole_inference(x)
            if out_size is not None:
                out = nn.functional.interpolate(
                    out, size=out_size, mode='bilinear', align_corners=False)
        return out

    def forward_ensemble(self, x, out_size=None):
        assert out_size is not None
        h, w = x.shape[-2:]
        # RETURNS SUM OF PROBABILITIES
        out_stack = 0
        cnt = 0
        for flip, scale in itertools.product([False, True], self.inference_ensemble_scale):
            new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
            if self.inference_ensemble_scale_divisor is not None:
                size_divisor = self.inference_ensemble_scale_divisor
                new_h = int(math.ceil(new_h / size_divisor)) * size_divisor
                new_w = int(math.ceil(new_w / size_divisor)) * size_divisor
            # this resize should mimic PIL
            x_inp = nn.functional.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
            x_inp = x_inp.flip(-1) if flip else x_inp
            if self.use_slide_inference:
                out = self.slide_inference(x_inp)
            else:
                out = self.whole_inference(x_inp)
            out = nn.functional.interpolate(
                out, size=out_size, mode='bilinear', align_corners=False)
            out = out.flip(-1) if flip else out
            out = nn.functional.softmax(out, dim=1)
            out_stack += out
            cnt += 1
        return out_stack / cnt

    def whole_inference(self, x):
        logits = self.head(self.backbone(x, img='trg', discriminate=True, inference=True)[0])
        logits = nn.functional.interpolate(
            logits, x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def slide_inference(self, img):
        """
        ---------------------------------------------------------------------------
        Copyright (c) OpenMMLab. All rights reserved.
        
        This source code is licensed under the license found in the
        LICENSE file in https://github.com/open-mmlab/mmsegmentation.
        ---------------------------------------------------------------------------
        """
        batch_size, _, h_img, w_img = img.size()
        h_stride, w_stride, h_crop, w_crop = self.get_inference_slide_params(
            h_img, w_img)
        num_classes = self.head.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.whole_inference(crop_img)
                preds += nn.functional.pad(crop_seg_logit,
                                            (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds

    def configure_optimizers(self):

        optimizer_seg = instantiate_class(
            self.optimizer_parameters_seg(), self.optimizer_init)
        optimizer_strainer = instantiate_class(
            self.optimizer_parameters_strainer(), self.optimizer_init)
        

        lr_scheduler = {
            "scheduler": instantiate_class(optimizer_seg, self.lr_scheduler_init),
            "interval": "step"
        }

        lr_scheduler_strainer = {
            "scheduler": instantiate_class(optimizer_strainer, self.lr_scheduler_init),
            "interval": "step"
        }

        optimizer_projector = torch.optim.Adamax([p for p in self.condition_projector.parameters() if p.requires_grad == True], lr=self.projector_lr)

        return [optimizer_seg, optimizer_strainer, optimizer_projector], [lr_scheduler, lr_scheduler_strainer]
    
    def optimizer_parameters_seg(self):
        proj_weight_params = []
        proj_bias_params = []
        other_weight_params = []
        other_bias_params = []
        strainer_params = []
        head_weight_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad: 
                continue
            if name.startswith('contrastive_head'):
                proj_bias_params.append(p) if len(p.shape) == 1 else proj_weight_params.append(p)
            elif 'strainer' in name:
                pass
            elif name.startswith('head'):
                head_weight_params.append(p)
            else:
                other_bias_params.append(p) if len(p.shape) == 1 else other_weight_params.append(p)
        lr = 1e-5
        weight_decay = 1e-2
        return [
            {'name': 'other_weight', 'params': other_weight_params,
                'lr': lr, 'weight_decay': weight_decay},
            {'name': 'other_bias', 'params': other_bias_params,
                'lr': lr, 'weight_decay': 0},
            {'name': 'proj_weight', 'params': proj_weight_params,
                'lr': self.projection_head_lr_factor * lr, 'weight_decay': weight_decay},
            {'name': 'proj_bias', 'params': proj_bias_params,
                'lr': self.projection_head_lr_factor * lr, 'weight_decay': 0},
            {'name': 'head_weight', 'params': head_weight_params,
                'lr': self.head_lr_factor * lr, 'weight_decay': weight_decay}
        ]

    def optimizer_parameters_strainer(self):
        proj_weight_params = []
        proj_bias_params = []
        other_weight_params = []
        other_bias_params = []
        strainer_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad: 
                continue
            if 'strainer' in name:
                strainer_params.append(p)
            else:
                pass
        lr = self.optimizer_init['init_args']['lr']
        weight_decay = self.optimizer_init['init_args']['weight_decay']
        return [
            {'name': 'strainer', 'params': strainer_params,
                'lr': lr, 'weight_decay': weight_decay} 
        ]

    @torch.no_grad()
    def align(self, logits_ref, images_ref, images_trg):
        assert self.alignment_head is not None

        h, w = logits_ref.shape[-2:]
        images_trg = nn.functional.interpolate(images_trg, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        images_ref = nn.functional.interpolate(images_ref, size=(h, w), mode='bilinear', align_corners=False, antialias=True)

        images_trg_256 = nn.functional.interpolate(
            images_trg, size=(256, 256), mode='area')
        images_ref_256 = nn.functional.interpolate(
            images_ref, size=(256, 256), mode='area')

        x_backbone = self.alignment_backbone(
            torch.cat([images_ref, images_trg]), extract_only_indices=[-3, -2])
        unpacked_x = [torch.tensor_split(l, 2) for l in x_backbone]
        pyr_ref, pyr_trg = zip(*unpacked_x)
        x_backbone_256 = self.alignment_backbone(
            torch.cat([images_ref_256, images_trg_256]), extract_only_indices=[-2, -1])
        unpacked_x_256 = [torch.tensor_split(l, 2) for l in x_backbone_256]
        pyr_ref_256, pyr_trg_256 = zip(*unpacked_x_256)

        trg_ref_flow, trg_ref_uncert = self.alignment_head(
            pyr_trg, pyr_ref, pyr_trg_256, pyr_ref_256, (h, w))[-1]
        trg_ref_flow = nn.functional.interpolate(
            trg_ref_flow, size=(h, w), mode='bilinear', align_corners=False)
        trg_ref_uncert = nn.functional.interpolate(
            trg_ref_uncert, size=(h, w), mode='bilinear', align_corners=False)

        trg_ref_cert = estimate_probability_of_confidence_interval(
            trg_ref_uncert, R=1.0)
        warped_ref_logits, trg_ref_mask = warp(
            logits_ref, trg_ref_flow, return_mask=True)
        warp_confidence = trg_ref_mask.unsqueeze(1) * trg_ref_cert
        return warped_ref_logits, warp_confidence

    @staticmethod
    def get_inference_slide_params(h, w):
        if h == w:
            return 1, 1, h, w
        if min(h, w) == h:  # wide image
            assert w <= 2 * h
            h_crop, w_crop, h_stride = h, h, 1
            w_stride = w // 2 - h // 2 if w > 1.5 * h else w - h
        else:  # tall image
            assert h <= 2 * w
            h_crop, w_crop, w_stride = w, w, 1
            h_stride = h // 2 - w // 2 if h > 1.5 * w else h - w
        return h_stride, w_stride, h_crop, w_crop

    @torch.no_grad()
    def update_momentum_encoder(self):
        m = min(1.0 - 1 / (float(self.global_step) + 1.0),
                self.ema_momentum)  # limit momentum in the beginning
        for param, param_m in zip(
                itertools.chain(self.backbone.parameters(), self.contrastive_head.parameters()),
                itertools.chain(self.m_backbone.parameters(), self.m_contrastive_head.parameters())):
            if not param.data.shape:
                param_m.data = param_m.data * m + param.data * (1. - m)
            else:
                param_m.data[:] = param_m[:].data[:] * \
                    m + param[:].data[:] * (1. - m)

    def train(self, mode=True):
        super().train(mode=mode)
        for m in filter(None, [self.alignment_backbone, self.alignment_head]):
            m.eval()
        for m in filter(None, [self.m_backbone, self.m_contrastive_head]):
            if isinstance(m, (nn.modules.dropout._DropoutNd, DropPath)):
                m.training = False


    @property
    def ckpt_dir(self):
        # mirroring https://github.com/Lightning-AI/lightning/blob/3bee81960a6f8979c8e1b5e747a17124feee652d/src/pytorch_lightning/callbacks/model_checkpoint.py#L571
        for cb in self.trainer.callbacks:
            if isinstance(cb, Checkpoint):
                if cb.dirpath is not None:
                    return cb.dirpath

        if len(self.trainer.loggers) > 0:
            if self.trainer.loggers[0].save_dir is not None:
                save_dir = self.trainer.loggers[0].save_dir
            else:
                save_dir = self.trainer.default_root_dir
            name = self.trainer.loggers[0].name
            version = self.trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"

            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(self.trainer.default_root_dir, "checkpoints")

        return ckpt_path
