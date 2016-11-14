import _init_paths

import argparse
import pprint
import time, os, sys
import os.path as osp

import numpy as np
import caffe
import cv2
from mpi4py import MPI

from datasets.factory import get_imdb
from utils import pickle, unpickle
from eval_utils import mpi_dispatch, mpi_collect


# mpi setup
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
if mpi_rank > 0:
    # disable print on other mpi processes
    sys.stdout = open(os.devnull, 'w')


def extract(net, im_name, rois, blob_name):
    if len(rois) == 0:
        return []

    im = cv2.imread(im_name)
    im = im.astype(np.float32, copy=True)
    im -= np.array([[[104, 117, 124]]])
    h, w = im.shape[:2]

    batch = np.empty((len(rois), 224, 224, 3), dtype=np.float32)
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = np.round(roi[:4]).astype(np.int32)
        x1 = min(w-1, max(0, x1))
        y1 = min(h-1, max(0, y1))
        x2 = min(w, max(x1+1, x2))
        y2 = min(h, max(y1+1, y2))
        crop_im = im[y1:y2, x1:x2, :]
        batch[i, ...] = cv2.resize(crop_im, (224, 224),
                                   interpolation=cv2.INTER_LINEAR)
    batch = batch.transpose(0, 3, 1, 2)

    net.blobs['data'].reshape(*(batch.shape))
    net.blobs['data'].data[...] = batch
    net.forward()
    return net.blobs[blob_name].data.copy()


def main(args):
    # parse gpus
    gpus = map(int, args.gpus.split(','))
    assert len(gpus) >= mpi_size, "Number of GPUs must be >= MPI size"
    gpu_id = gpus[mpi_rank]

    # load imdb
    imdb = get_imdb(args.imdb_name)

    if args.eval_only:
        def _load(fname):
            fpath = osp.join(args.output_dir, fname)
            assert osp.isfile(fpath), "Must have extracted detections and " \
                                      "features first before evaluation"
            return unpickle(fpath)
        if mpi_rank == 0:
            gboxes = _load('gallery_detections.pkl')
            gfeatures = _load('gallery_features.pkl')
            pfeatures = _load('probe_features.pkl')
    else:
        # setup caffe
        caffe.mpi_init()
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

        # 1. Detect and extract features from all the gallery images in the imdb
        gboxes = unpickle(args.det_pickle)
        assert len(gboxes) == len(imdb.image_index)
        start, end = mpi_dispatch(len(imdb.image_index), mpi_size, mpi_rank)
        gfeatures = []
        for i in xrange(start, end):
            feat = extract(net, imdb.image_path_at(start + i), gboxes[i],
                           args.blob_name)
            gfeatures.append(feat)
            print 'gallery: {} / {}'.format(i-start+1, end-start)
        gfeatures = mpi_collect(mpi_comm, mpi_rank, gfeatures)

        # 2. Only extract features from given probe rois
        start, end = mpi_dispatch(len(imdb.probes), mpi_size, mpi_rank)
        pfeatures = []
        for i in xrange(start, end):
            im_name, roi = imdb.probes[start + i]
            roi = roi.reshape(1, 4)
            feat = extract(net, im_name, roi, args.blob_name)
            assert feat.shape[0] == 1
            pfeatures.append(feat[0])
            print 'probe: {} / {}'.format(i-start+1, end-start)
        pfeatures = mpi_collect(mpi_comm, mpi_rank, pfeatures)
        del net

        # Save
        if mpi_rank == 0:
            if not osp.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            pickle(gboxes, osp.join(args.output_dir, 'gallery_detections.pkl'))
            pickle(gfeatures, osp.join(args.output_dir, 'gallery_features.pkl'))
            pickle(pfeatures, osp.join(args.output_dir, 'probe_features.pkl'))

    # Evaluate
    if mpi_rank == 0:
        imdb.evaluate_detections(gboxes, det_thresh=args.det_thresh)
        imdb.evaluate_detections(gboxes, det_thresh=args.det_thresh,
                                 labeled_only=True)
        imdb.evaluate_search(gboxes, gfeatures, pfeatures,
             det_thresh=args.det_thresh,
             gallery_size=args.gallery_size,
             dump_json=osp.join(args.output_dir, 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_pickle',
                        required=True)
    parser.add_argument('--output_dir',
                        required=True)
    parser.add_argument('--gpus',
                        help='comma separated GPU device ids',
                        default='0')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='psdb_test')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network')
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test')
    parser.add_argument('--blob_name',
                        default='feat')
    parser.add_argument('--det_thresh',
                        help="detection score threshold to be evaluated",
                        type=float, default=0)
    parser.add_argument('--gallery_size',
                        help='gallery size for evaluation, -1 for full set',
                        type=int, default=100,
                        choices=[-1, 50, 100, 500, 1000, 2000, 4000])
    parser.add_argument('--eval_only',
                        help='skip the feature extraction and only do eval',
                        action='store_true')

    args = parser.parse_args()
    main(args)
