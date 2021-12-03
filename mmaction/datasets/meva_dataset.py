import copy
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

import mmcv
import numpy as np
from mmcv.utils import print_log
import json

from ..core.evaluation.ava_utils import ava_eval, read_labelmap, results2csv
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MEVADataset(BaseDataset):
    """AVA dataset for spatial temporal detection.

    Based on official AVA annotation files, the dataset loads raw frames,
    bounding boxes, proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Default: None.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Default: None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used. Default: 0.9.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used.
        num_classes (int): The number of classes of the dataset. Default: 81.
            (AVA has 80 action classes, another 1-dim is added for potential
            usage)
        custom_classes (list[int]): A subset of class ids from origin dataset.
            Please note that 0 should NOT be selected, and ``num_classes``
            should be equal to ``len(custom_classes) + 1``
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                        Default: 'RGB'.
        num_max_proposals (int): Max proposals number to store. Default: 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website. Default: 902.
        timestamp_end (int): The end point of included timestamps. The
            default value is referred from the official website. Default: 1798.
    """

    _FPS = 30

    def __init__(self,
                 ann_file,
                 pipeline,
                 label_file=None,
                 filename_tmpl='frame_{:05}.jpg', #'img_{:05}.jpg',
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_classes=81,
                 custom_classes=None,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 num_max_proposals=1000,
                 timestamp_start=900,
                 timestamp_end=1800):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`
        self.custom_classes = custom_classes
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = tuple([0] + custom_classes)
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.num_classes = num_classes
        self.filename_tmpl = filename_tmpl
        self.num_max_proposals = num_max_proposals
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.logger = get_root_logger()
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            modality=modality,
            num_classes=num_classes)

        if self.proposal_file is not None:
            self.proposals = mmcv.load(self.proposal_file)
        else:
            self.proposals = None

        '''
        if not test_mode:
            valid_indexes = self.filter_exclude_file()
            self.logger.info(
                f'{len(valid_indexes)} out of {len(self.video_infos)} '
                f'frames are valid.')
            self.video_infos = [self.video_infos[i] for i in valid_indexes]
        '''
        
        

    def parse_img_record(self, img_records):
        """Merge image records of the same entity at the same time.

        Args:
            img_records (list[dict]): List of img_records (lines in AVA
                annotations).

        Returns:
            tuple(list): A tuple consists of lists of bboxes, action labels and
                entity_ids
        """
        bboxes, labels, entity_ids = [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)
            selected_records = list(
                filter(
                    lambda x: np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            num_selected_records = len(selected_records)
            img_records = list(
                filter(
                    lambda x: not np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def filter_exclude_file(self):
        """Filter out records in the exclude_file."""
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.video_infos)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, video_info in enumerate(self.video_infos):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (video_info['video_id'] == video_id
                            and video_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        return valid_indexes

    def load_annotations(self):
        """Load MEVA annotations."""
        """
        {
            'videoname':***,
            'frame_id':***,
            'duration':32,
            'bbox_clip':[x,y,x,y],
            'bbox_object':[x,y,x,y] or None,
            'label':*
        }
        """
        self.dict_util = {
            "person_picks_up_object": 1,
            "person_puts_down_object": 2,
        }
        # self.dict_util = {
        #     "person_stands_up": 0,
        #     "person_sits_down": 1,
        #     "person_texts_on_phone": 2,
        #     "person": 3
        # }
        video_infos = []
        contents = json.load(open(self.ann_file, 'r'))
        
        for content in contents:
            
            videoname = content['videoname'].replace('.r13', '')
            frame_dir = videoname
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            frame_id = int(content['frame_id'])
            duration = int(content['duration'])
            bbox_clip = content['bbox_clip']
            bbox_object = content['bbox_object']
            w = bbox_clip[2] - bbox_clip[0]
            h = bbox_clip[3] - bbox_clip[1]
            gt_bboxes = [[(bbox_object[0] - bbox_clip[0]) / w, (bbox_object[1] - bbox_clip[1]) / h, (bbox_object[2] - bbox_clip[0]) / w, (bbox_object[3] - bbox_clip[1]) / h]]
            # gt_bboxes = [[bbox_object[0] - bbox_clip[0], bbox_object[1] - bbox_clip[1], bbox_object[2] - bbox_clip[0], bbox_object[3] - bbox_clip[1]]]
            gt_labels = [[self.dict_util[content['label']], ]]
            assert gt_bboxes[0][1] < gt_bboxes[0][3] and gt_bboxes[0][0] < gt_bboxes[0][2]
            video_info = dict(
                fid=frame_id,
                frame_dir=frame_dir,
                bbox_clip=bbox_clip,
                duration=int(duration),
                gt_bboxes = gt_bboxes,
                gt_labels=gt_labels)
            video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['proposals'] = np.array([[0, 0, 1, 1]], dtype=np.float32)
        results['scores'] = np.array([1], dtype=np.float32)
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['proposals'] = np.array([[0, 0, 1, 1]], dtype=np.float32)
        results['scores'] = np.array([1], dtype=np.float32)
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality

        return self.pipeline(results)

    def dump_results(self, results, out):
        """Dump predictions into a csv file."""
        assert out.endswith('csv')
        results2csv(self, results, out, self.custom_classes)

    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):
        """Evaluate the prediction results and report mAP."""
        assert len(metrics) == 1 and metrics[0] == 'mAP', (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            'See https://github.com/open-mmlab/mmaction2/pull/567 '
            'for more info.')
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'AVA_{time_now}_result.csv'
        results2csv(self, results, temp_file, self.custom_classes)

        ret = {}
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            eval_result = ava_eval(
                temp_file,
                metric,
                self.label_file,
                self.ann_file,
                self.exclude_file,
                custom_classes=self.custom_classes)
            log_msg = []
            for k, v in eval_result.items():
                log_msg.append(f'\n{k}\t{v: .4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)

        os.remove(temp_file)

        return ret
