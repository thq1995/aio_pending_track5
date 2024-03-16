"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np

import time
import argparse
from filterpy.kalman import KalmanFilter
import copy

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.id_det_rect = dict()
        # 每个tracker id 对应的按detection检测的类别序列
        # {tid: [(frame, class, dect[cx, cy, w, h], score), ..., ]}
        self.id_det_class = dict()

    def enlarge_rect(self, dets):
        for i in range(dets.shape[0]):
            x0, y0, x1, y1, score = dets[i]
            if x1 - x0 < 200 and y1 - y0 < 200:
                dets[i][0] -= 30
                dets[i][1] -= 30
                dets[i][2] += 30
                dets[i][3] += 30
        return dets

    def update(self, dets=np.empty((0, 5)), frame=0):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        dets_bak = copy.deepcopy(dets)
        # convert to [x1,y1,x2,y2] to [cx,cy,w,h]
        # dets_bak[:, 2:4] = dets_bak[:, 2:4] - dets_bak[:, 0:2]
        # dets_bak[:, 0:2] += dets_bak[:, 2:4] / 2
        # dets = self.enlarge_rect(dets)

        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        ret_bak = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            del self.id_det_rect[self.trackers[t].id]
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.id_det_rect[self.trackers[m[1]].id] = dets_bak[m[0], :]  # 记录检测框
            #
            self.id_det_class[self.trackers[m[1]].id].append(
                (frame, int(dets_bak[m[0], -1]), dets_bak[m[0], :4], dets_bak[m[0], -2]))

            self.id_det_rect[self.trackers[m[1]].id][-1] = self.trackers[m[1]].id

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            class_idx = dets[i, -1]
            trk = KalmanBoxTracker(dets[i, :-1])
            self.trackers.append(trk)
            if trk.id not in self.id_det_rect:
                self.id_det_rect[trk.id] = dets_bak[i, :]
                #
                if trk.id not in self.id_det_class:
                    self.id_det_class[trk.id] = [(frame, int(dets_bak[i, -1]), dets_bak[i, :4], dets_bak[i, -2])]
                else:
                    self.id_det_class[trk.id].append((frame, int(dets_bak[i, -1]), dets_bak[i, :4], dets_bak[i, -2]))
                #
                self.id_det_rect[trk.id][-1] = trk.id

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                rect_bak = np.asarray(self.id_det_rect[trk.id]).reshape(1, -1)
                ret_bak.append(rect_bak)
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                del self.id_det_rect[self.trackers[i].id]
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret), np.concatenate(ret_bak)
        return np.empty((0, 5)), np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", default='data/submit.txt',help="Path to detections.", type=str)
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=10)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=0)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seq_path = args.seq_path

    total_time = 0.0
    total_frames = 0
    refined_label_path ="../submit.txt"
    video_seq = {}

    with open(seq_path, 'r') as f:
        for line in f:
            video_id, *dt_info = line.rstrip('\n').split(',')
            video_id = int(video_id)
            if video_id in video_seq:
                video_seq[video_id].append(dt_info)
            else:
                video_seq[video_id] = [dt_info]
    f.close()
    cnt = 0
    with open(refined_label_path, 'w') as out_fn:
        for video_id in sorted(video_seq):
            mot_tracker = Sort(max_age=args.max_age,
                               min_hits=args.min_hits,
                               iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
            print("Processing video %s." % video_id)
            for seq_dets_data in sorted(video_seq[video_id]):
                frame = int(seq_dets_data[0])
                seq_dets_data = list(map(int, seq_dets_data[1:-1])) + [float(seq_dets_data[-1])]
                seq_dets = np.array(seq_dets_data)

                if len(seq_dets.shape) == 1:
                    seq_dets = np.array([seq_dets])
                seq_dets[:, [-2, -1]] = seq_dets[:, [-1, -2]]
                dets = seq_dets  # cx cy w h s cls
                # filter bboxes of score < 0.3
                #dets = dets[(dets[:, -2] > 0.3), :]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

                total_frames += 1
                start_time = time.time()
                trackers, _ = mot_tracker.update(dets, frame)  # x1,y1,x2,y2, s,cls
                cycle_time = time.time() - start_time
                total_time += cycle_time

            refined_id_det_class = {}
            for tid, track_list in mot_tracker.id_det_class.items():
                idx_class_cnt = {}
                # Count the number of occurrences of different categories in the same ID
                for i in track_list:
                    frame, class_idx, rect, score = i
                    if class_idx in idx_class_cnt:
                        idx_class_cnt[class_idx] += 1
                    else:
                        idx_class_cnt[class_idx] = 1
                # Replace ID
                if len(idx_class_cnt.keys()) > 1:
                    if max(idx_class_cnt.values()) > sum(idx_class_cnt.values()) / 2:
                        count2classid = {k: v for v, k in idx_class_cnt.items()}
                        max_classid = count2classid[max(idx_class_cnt.values())]
                        refined_track_list = []
                        for i in track_list:
                            frame_, class_idx_, rect_, score_ = i
                            if class_idx_ != max_classid:
                                # frame, update_class, bbox, score
                                refined_track_list.append((frame_, max_classid, rect_, score_))
                            else:
                                refined_track_list.append(i)
                        refined_id_det_class[tid] = refined_track_list
                    else:
                        refined_id_det_class[tid] = track_list
                else:
                    refined_id_det_class[tid] = track_list
                # print(track_list)
                print("before：idx_class_cnt ", idx_class_cnt)
                idx_class_cnt.clear()
                for i in refined_id_det_class[tid]:
                    frame, class_idx, rect, score = i
                    if class_idx in idx_class_cnt:
                        idx_class_cnt[class_idx] += 1
                    else:
                        idx_class_cnt[class_idx] = 1
                print("after：idx_class_cnt ", idx_class_cnt)
            # save txt
            frame_dets = {}
            for id, track_list in refined_id_det_class.items():
                for i in track_list:
                    frame_, class_idx_, rect_, score_ = i
                    if len(rect_.shape) == 1:
                        rect_ = np.array([rect_])

                    # convert to [x1,y1,x2,y2] to [x1,y1,w,h]
                    rect_[:, 2:4] -= rect_[:, 0:2]

                    rect_ = [str("%d" % i) for i in rect_[0]]
                    line = [str(video_id)] + [str(frame_)] + rect_ + [str(class_idx_)] + [str(score_)]
                    if frame_ not in frame_dets:
                        frame_dets[frame_] = [','.join(line) + '\n']
                    else:
                        frame_dets[frame_].append(','.join(line) + '\n')

            for frame, dets_list in frame_dets.items():
                cnt += len(dets_list)
                out_fn.writelines(dets_list)

            print(cnt)

        print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
            total_time, total_frames, total_frames / total_time))
    out_fn.close()


if __name__ == '__main__':
    main()