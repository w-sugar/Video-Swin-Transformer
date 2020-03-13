import os
import os.path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Non-Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Attributes:
        dataloader (DataLoader): Evaluation data loader.
        interval (int): Epoch interval for evaluation. Default: 1.
        eval_kwargs (option): Arguments for evaluation.
    """

    def __init__(self, data_loader, interval=1, **eval_kwargs):
        if isinstance(data_loader, DataLoader):
            self.data_loader = data_loader
        else:
            raise TypeError(
                f'data_loader must be a DataLoader, not {type(data_loader)}')
        self.dataset = data_loader.dataset
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        # compute output
        for data in self.data_loader:
            with torch.no_grad():
                output = runner.model(return_loss=False, **data)
            results.extend(output)

            batch_size = data['imgs'].size(0)
            for _ in range(batch_size):
                prog_bar.update()
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        results = [res.squeeze() for res in results]
        eval_res = self.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.
    """

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(return_loss=False, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        # store results in temporal pickle for each worker.
        tmp_file = osp.join(runner.work_dir, f'temp_{runner.rank}.pkl')
        mmcv.dump(results, tmp_file)
        dist.barrier()

        # collect all outputs and eval the results.
        if runner.rank == 0:
            print('\n')
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, f'temp_{i}.pkl')
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
            os.remove(osp.join(runner.work_dir, 'temp_0.pkl'))

        return