import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from typing import Dict, List, Optional, Tuple, Union
from mmdet.registry import MODELS


class preset_text_prompts_generator(BaseModule):
    """ text prompts memory.

    Args:

    """

    def __init__(self,
                 in_channels: int,
                 RoI_size: int = 2,
                 trainable: bool = False,
                 with_avg_pool: bool = True,
                 num_branch_fcs: int = 2,
                 fc_out_channels: int = 1024,
                 fc_loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.1),
                 save_prompts: list = None,
                 save_prompt_path=''):
        super().__init__()
        self.trainable = trainable
        if self.trainable is True:
            self.save_prompts = save_prompts
            torch.save(self.save_prompts, save_prompt_path)
        else:
            self.save_prompts = torch.load(save_prompt_path)
        self.prompt_len = len(self.save_prompts)

        self.with_avg_pool = with_avg_pool
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((RoI_size, RoI_size))
        self.fc_activation = torch.nn.ReLU()
        self.cls_fcs = nn.ModuleList()
        self.fc_out_channels = fc_out_channels
        self.fc_in_channels = in_channels * RoI_size * RoI_size
        for i in range(num_branch_fcs):
            fc_in_channels = (
                self.fc_in_channels if i == 0 else self.fc_out_channels)
            self.cls_fcs.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))
        self.fc_cls = nn.Linear(self.fc_out_channels, self.prompt_len)

        self.loss_cls = MODELS.build(fc_loss_cls)

    def update_prompts_embedding(self, save_prompts_embedding):
        self.save_prompts_embedding = save_prompts_embedding

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,

        if pre_logits.dim() > 2:
            if self.with_avg_pool:
                pre_logits = self.avg_pool(pre_logits)
            pre_logits = pre_logits.flatten(1)
        for fc in self.cls_fcs:
            pre_logits = self.fc_activation(fc(pre_logits))
        pre_logits = self.fc_cls(pre_logits)

        return pre_logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List,
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  batch_data_samples: List, **kwargs):
        """Perform forward propagation and loss calculation of the detection
                head on the queries of the upstream network.

                Args:
                    hidden_states (Tensor): Hidden states output from each decoder
                        layer, has shape (num_decoder_layers, bs, num_queries_total,
                        dim), where `num_queries_total` is the sum of
                        `num_denoising_queries` and `num_matching_queries` when
                        `self.training` is `True`, else `num_matching_queries`.
                    batch_data_samples (list[:obj:`DetDataSample`]): The Data
                        Samples. It usually includes information such as
                        `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

                Returns:
                    dict: A dictionary of loss components.
                """
        # batch_gt_instances = []
        # batch_img_metas = []
        losses = dict()
        # for data_sample in batch_data_samples:
        #     batch_img_metas.append(data_sample.metainfo)
        #     batch_gt_instances.append(data_sample.gt_instances)

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        gt_targets = torch.zeros(cls_score.shape, dtype=cls_score.dtype, device=cls_score.device)
        # remove redundent
        for gt_label, gt_target in zip(gt_labels, gt_targets):
            if len(gt_label) > 0:
                gt_label = torch.unique(gt_label)
                gt_target[gt_label] = 1.0

        losses_multi_cls = self.loss_cls(cls_score, gt_targets)

        # losses_multi_cls = self.loss_by_feat(*loss_inputs)

        losses['multi_cls_loss'] = losses_multi_cls
        return losses


    def predict(
            self,
            feats: Tuple[torch.Tensor],
            data_samples=None
    ) -> List:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score)
        return predictions

    def _get_predictions(self, cls_score):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        text_prompts_list_all = []
        text_prompts_list = []
        with torch.no_grad():
            # 1. get index
            # pred_scores = torch.nn.functional.sigmoid(cls_score)
            pred_scores = cls_score
            for pred_score in pred_scores:
                pred_score_mean = torch.mean(pred_score, dim=-1)
                pred_score_var = torch.std(pred_score, dim=-1, unbiased=False)
                pos_neg_thresh = pred_score_mean - pred_score_var
                pred_score_pos = pred_score[pred_score.ge(pos_neg_thresh)]
                pred_score_neg = pred_score[pred_score.le(pos_neg_thresh)]
                if len(pred_score_pos) > 1:
                    pred_score_pos_mean = torch.mean(pred_score_pos, dim=-1)
                    pred_score_pos_var = torch.std(pred_score_pos, dim=-1, unbiased=False)
                else:
                    pred_score_pos_mean = pred_score_mean
                    pred_score_pos_var = pred_score_var
                if len(pred_score_neg) > 0:
                    pred_score_neg_mean = torch.mean(pred_score_neg, dim=-1)
                    pred_score_neg_var = torch.std(pred_score_neg, dim=-1, unbiased=False)
                else:
                    pred_score_neg_mean = torch.tensor(0.0)
                    pred_score_neg_var = torch.tensor(0.0)

                if (pred_score_pos_mean - 3 * pred_score_pos_var) > (pred_score_neg_mean + 3 * pred_score_neg_var):
                    pred_score_tresh = pred_score_neg_mean + 3 * pred_score_neg_var
                else:
                    pred_score_tresh = pred_score_pos_mean - 3 * pred_score_pos_var

                pred_idx = torch.nonzero(pred_score.gt(pred_score_tresh))

                if len(pred_idx) < 3:
                    _, pred_idx = torch.topk(pred_score, 3, dim=-1)

                pred_idx = pred_idx.sort().values
                # 2. get prompts
                for pred_item in pred_idx:
                    search_prompt = self.save_prompts[pred_item]
                    if search_prompt in text_prompts_list:
                        continue
                    else:
                        text_prompts_list.append(search_prompt)
            for i in range(len(pred_scores)):
                text_prompts_list_all.append(text_prompts_list)

        return text_prompts_list_all
