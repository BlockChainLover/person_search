import _init_paths

import argparse
import time, os, sys
import os.path as osp

import numpy as np
from scipy.io import loadmat

from datasets.factory import get_imdb
from utils import pickle, unpickle


def _load_data(input_dir, det_format, transpose_feat):
    # gallery detection: n_det x [x1, y1, x2, y2, score] per image
    gdet = loadmat(osp.join(input_dir, 'gal_det.mat'))['gal_det'].squeeze()
    if det_format == 'xywh':
        for d in gdet:
            d[:, 2:4] += d[:, 0:2]

    # gallery features: n_det x D features per image
    gfeat = loadmat(osp.join(input_dir, 'gal_feat.mat'))['gal_feat'].squeeze()
    if transpose_feat:
        for i in xrange(len(gfeat)):
            gfeat[i] = gfeat[i].T

    # probe features: D dimensional features per image
    pfeat = loadmat(osp.join(input_dir, 'prb_feat.mat'))['prb_feat']
    if transpose_feat:
        pfeat = pfeat.T

    return gdet, gfeat, pfeat


def main(args):
    gdet, gfeat, pfeat = _load_data(args.input_dir,
                                    args.det_format, args.transpose_feat)
    imdb = get_imdb(args.imdb_name)
    imdb.evaluate_detections(gdet,
                             det_thresh=args.det_thresh,
                             iou_thresh=args.iou_thresh)
    imdb.evaluate_detections(gdet,
                             det_thresh=args.det_thresh,
                             iou_thresh=args.iou_thresh,
                             labeled_only=True)
    imdb.evaluate_search(gdet, gfeat, pfeat,
                         det_thresh=args.det_thresh,
                         iou_thresh=args.iou_thresh,
                         gallery_size=args.gallery_size,
                         dump_json=osp.join(args.input_dir, 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',
                        help="Directory containing prb and gal mat files")
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='psdb_test')
    parser.add_argument('--det_thresh',
                        help="detection score threshold to be evaluated",
                        type=float, default=0)
    parser.add_argument('--iou_thresh',
                        help="IoU threshold to be evaluated",
                        type=float, default=0.5)
    parser.add_argument('--gallery_size',
                        help='gallery size for evaluation, -1 for full set',
                        type=int, default=100,
                        choices=[-1, 50, 100, 500, 1000, 2000, 4000])
    parser.add_argument('--det_format',
                        help="Detection result format",
                        choices=['xywh', 'x1y1x2y2'],
                        default='x1y1x2y2')
    parser.add_argument('--transpose_feat',
                        action='store_true')
    args = parser.parse_args()
    main(args)
