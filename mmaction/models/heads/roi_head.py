import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.roi_heads import StandardRoIHead
    from mmdet.models.dense_heads.rpn_head import RPNHead
    from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class AVARPNHead(RPNHead):

        def forward_single(self, x):
            """Forward feature map of a single scale level."""
            x = x[:, :, 0]
            x = self.rpn_conv(x)
            x = F.relu(x, inplace=True)
            rpn_cls_score = self.rpn_cls(x)
            rpn_bbox_pred = self.rpn_reg(x)
            return rpn_cls_score, rpn_bbox_pred

        def loss(self,
                cls_scores,
                bbox_preds,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore=None):
            """Compute losses of the head.

            Args:
                cls_scores (list[Tensor]): Box scores for each scale level
                    Has shape (N, num_anchors * num_classes, H, W)
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level with shape (N, num_anchors * 4, H, W)
                gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                    shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                img_metas (list[dict]): Meta information of each image, e.g.,
                    image size, scaling factor, etc.
                gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                    boxes can be ignored when computing the loss.

            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            """
            # losses = super(RPNHead, self).loss(
            losses = self.loss_1(
                cls_scores,
                bbox_preds,
                gt_bboxes,
                None,
                img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
            return dict(
                loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

        @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
        def loss_1(self,
                cls_scores,
                bbox_preds,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore=None):
            """Compute losses of the head.

            Args:
                cls_scores (list[Tensor]): Box scores for each scale level
                    Has shape (N, num_anchors * num_classes, H, W)
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level with shape (N, num_anchors * 4, H, W)
                gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                    shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): class indices corresponding to each box
                img_metas (list[dict]): Meta information of each image, e.g.,
                    image size, scaling factor, etc.
                gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                    boxes can be ignored when computing the loss. Default: None

            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            """
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            assert len(featmap_sizes) == self.anchor_generator.num_levels

            device = cls_scores[0].device

            anchor_list, valid_flag_list = self.get_anchors(
                featmap_sizes, img_metas, device=device)
            label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
            cls_reg_targets = self.get_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                label_channels=label_channels)
            if cls_reg_targets is None:
                return None
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
            num_total_samples = (
                num_total_pos + num_total_neg if self.sampling else num_total_pos)

            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # concat all level anchors and flags to a single tensor
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list,
                                            num_level_anchors)

            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples)
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                        bbox_targets, bbox_weights, num_total_samples):
            """Compute loss of a single scale level.

            Args:
                cls_score (Tensor): Box scores for each scale level
                    Has shape (N, num_anchors * num_classes, H, W).
                bbox_pred (Tensor): Box energies / deltas for each scale
                    level with shape (N, num_anchors * 4, H, W).
                anchors (Tensor): Box reference for each scale level with shape
                    (N, num_total_anchors, 4).
                labels (Tensor): Labels of each anchors with shape
                    (N, num_total_anchors).
                label_weights (Tensor): Label weights of each anchor with shape
                    (N, num_total_anchors)
                bbox_targets (Tensor): BBox regression targets of each anchor wight
                    shape (N, num_total_anchors, 4).
                bbox_weights (Tensor): BBox regression loss weights of each anchor
                    with shape (N, num_total_anchors, 4).
                num_total_samples (int): If sampling, num total samples equal to
                    the number of total anchors; Otherwise, it is the number of
                    positive anchors.

            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            """
            # classification loss
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
            # regression loss
            bbox_targets = bbox_targets.reshape(-1, 4)
            bbox_weights = bbox_weights.reshape(-1, 4)  
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            if self.reg_decoded_bbox:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, it
                # decodes the already encoded coordinates to absolute format.
                anchors = anchors.reshape(-1, 4)
                bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)
            return loss_cls, loss_bbox
        def _get_targets_single(self,
                                flat_anchors,
                                valid_flags,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                img_meta,
                                label_channels=1,
                                unmap_outputs=True):
            """Compute regression and classification targets for anchors in a
            single image.

            Args:
                flat_anchors (Tensor): Multi-level anchors of the image, which are
                    concatenated into a single tensor of shape (num_anchors ,4)
                valid_flags (Tensor): Multi level valid flags of the image,
                    which are concatenated into a single tensor of
                        shape (num_anchors,).
                gt_bboxes (Tensor): Ground truth bboxes of the image,
                    shape (num_gts, 4).
                gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                    ignored, shape (num_ignored_gts, 4).
                img_meta (dict): Meta info of the image.
                gt_labels (Tensor): Ground truth labels of each box,
                    shape (num_gts,).
                label_channels (int): Channel of label.
                unmap_outputs (bool): Whether to map outputs back to the original
                    set of anchors.

            Returns:
                tuple:
                    labels_list (list[Tensor]): Labels of each level
                    label_weights_list (list[Tensor]): Label weights of each level
                    bbox_targets_list (list[Tensor]): BBox targets of each level
                    bbox_weights_list (list[Tensor]): BBox weights of each level
                    num_total_pos (int): Number of positive samples in all images
                    num_total_neg (int): Number of negative samples in all images
            """
            inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                            img_meta['img_shape'][:2],
                                            self.train_cfg.allowed_border)
            if not inside_flags.any():
                return (None, ) * 7
            # assign gt and sample anchors
            anchors = flat_anchors[inside_flags, :]

            print(anchors, gt_bboxes, gt_bboxes_ignore)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
            sampling_result = self.sampler.sample(assign_result, anchors,
                                                gt_bboxes)

            num_valid_anchors = anchors.shape[0]
            bbox_targets = torch.zeros_like(anchors)
            bbox_weights = torch.zeros_like(anchors)
            labels = anchors.new_full((num_valid_anchors, ),
                                    self.num_classes,
                                    dtype=torch.long)
            label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
            if len(pos_inds) > 0:
                if not self.reg_decoded_bbox:
                    pos_bbox_targets = self.bbox_coder.encode(
                        sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
                else:
                    pos_bbox_targets = sampling_result.pos_gt_bboxes
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_weights[pos_inds, :] = 1.0
                if gt_labels is None:
                    # Only rpn gives gt_labels as None
                    # Foreground is the first class since v2.5.0
                    labels[pos_inds] = 0
                else:
                    labels[pos_inds] = gt_labels[
                        sampling_result.pos_assigned_gt_inds]
                if self.train_cfg.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.train_cfg.pos_weight
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            # map up to original set of anchors
            if unmap_outputs:
                num_total_anchors = flat_anchors.size(0)
                labels = unmap(
                    labels, num_total_anchors, inside_flags,
                    fill=self.num_classes)  # fill bg label
                label_weights = unmap(label_weights, num_total_anchors,
                                    inside_flags)
                bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
                bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                    neg_inds, sampling_result)

    @MMDET_HEADS.register_module()
    class AVARoIHead(StandardRoIHead):

        def _bbox_forward(self, x, rois, img_metas):
            """Defines the computation performed to get bbox predictions.

            Args:
                x (torch.Tensor): The input tensor.
                rois (torch.Tensor): The regions of interest.
                img_metas (list): The meta info of images

            Returns:
                dict: bbox predictions with features and classification scores.
            """
            bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)

            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=img_metas)

            cls_score, bbox_pred = self.bbox_head(bbox_feat)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
                                gt_labels, img_metas):
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois, img_metas)

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results

        def simple_test(self,
                        x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            """Defines the computation performed for simple testing."""
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois, img_metas)
            cls_score = bbox_results['cls_score']

            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels
else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class AVARoIHead:
        pass
