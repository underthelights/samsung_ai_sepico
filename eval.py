# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import pandas as pd

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--palette',
        action='store_true',
        help='Whether to use palette in format.'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # cfg = mmcv.Config.fromfile(args.config)
    cfg = mmcv.Config.fromfile('/content/drive/MyDrive/Colab_Notebooks/samsung_ai/open/SePiCo/work_dirs/local-exp1/230921_0925_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_flat2fisheye_seed76_3a066/230921_0925_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_flat2fisheye_seed76_3a066.json')

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
    test_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(
          type='MultiScaleFlipAug',
          # img_scale=(960, 540),
          img_scale=(1920, 1080),
          # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
          # img_ratios=[(0.5)],
          flip=False,
          transforms=[
              dict(type='Resize', img_scale=(960, 540), keep_ratio=True),
              dict(type='RandomFlip'),
              dict(type='Normalize', **img_norm_cfg),
              dict(type='ImageToTensor', keys=['img']),
              dict(type='Collect', keys=['img']),
      ])
    ]

    # test_pipeline = [
    #   dict(type='LoadImageFromFile'),
    #   dict(
    #       type='MultiScaleFlipAug',
    #       img_scale=(1920, 960),
    #       # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #       flip=False,
    #       transforms=[
    #           dict(type='Resize', keep_ratio=True),
    #           # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
    #           # dict(type='ResizeToMultiple', size_divisor=8),
    #           dict(type='RandomFlip'),
    #           dict(type='Normalize', **img_norm_cfg),
    #           dict(type='ImageToTensor', keys=['img']),
    #           dict(type='Collect', keys=['img']),
    #       ])
    # ]

    cfg.data.test = dict(
        type='FisheyeDataset',
        data_root='../',
        img_dir='test_image',
        pipeline=test_pipeline
      )
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        '/content/drive/MyDrive/Colab_Notebooks/samsung_ai/open/SePiCo/work_dirs/local-exp1/230921_0925_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_flat2fisheye_seed76_3a066/iter_20000.pth',
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    submit = pd.read_csv('../sample_submission.csv')
    submit['mask_rle'] = outputs
    submit.to_csv('./sepico_submit.csv', index=False)

    # rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         mmcv.dump(outputs, args.out)
    #     kwargs = {} if args.eval_options is None else args.eval_options
    #     if args.format_only:
    #         dataset.format_results(outputs, do_palette=args.palette, cmap=model.module.PALETTE, **kwargs)
    #     if args.eval:
    #         dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
