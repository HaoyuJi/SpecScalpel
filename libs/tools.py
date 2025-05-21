import numpy as np
import torch

import torch.nn.functional as F
import torch.nn as nn


def segment_video_labels(video_labels_batch):
    segments_batch = []

    for video_labels in video_labels_batch:
        segments = []
        current_label = None
        segment_start = 0
        for i, label in enumerate(video_labels):
            label = label.item()
            if label != current_label:
                if (current_label is not None) and (current_label !=255):
                    segment_length = i - segment_start
                    segments.append((current_label, segment_length, segment_start, i - 1))
                current_label = label
                segment_start = i

        if (current_label is not None) and (video_labels[-1] !=255):
            segment_length = len(video_labels) - segment_start
            segments.append((current_label, segment_length, segment_start, len(video_labels) - 1))
        segments_batch.append(segments)

    return segments_batch


def split_mixed_class(segments_batch, count):
    labels = []
    start_ends_batch = []
    for segments in segments_batch:
        # labels = []
        start_ends = []
        for i in range(len(segments)-count+1):
            label = [segments[j][0] for j in range(i, i+count)]
            start_end = [segments[i][2], segments[i+count-1][3]]
            labels.append(label)
            start_ends.append(start_end)

        start_ends_batch.append(start_ends)

    return labels, start_ends_batch


def split_feature(feature, device, num_segments=16):


    N, C, T = feature.size()

    segment_length = T // num_segments

    features_split = []

    for i in range(N):
        for j in range(num_segments-1):
            split_feature = feature[i, :, j*segment_length:(j+1)*segment_length].mean(dim=-1)
            features_split.append(split_feature)
        features_split.append(feature[i, :, (num_segments-1)*segment_length:-1].mean(dim=-1))

    features_split = torch.stack(features_split).to(device)

    return features_split

def split_gt(gt, device, num_segments=16):


    N, T = gt.size()

    segment_length = T // num_segments

    gts_split = []

    for i in range(N):
        for j in range(num_segments-1):
            split_gt = gt[i, j*segment_length:(j+1)*segment_length]
            gt_split = []
            last_element = split_gt[0]
            gt_label = [last_element]
            for element in split_gt:
                if (element != last_element) and (element != 255):
                    gt_label.append(element)
                    last_element = element

            gt_split = torch.stack(gt_label).to(device)
            if (gt_split == 255).any().item() == False:
                gts_split.append(gt_split)


        split_gt = gt[i, (num_segments-1)*segment_length:-1]
        gt_split = []
        last_element = split_gt[0]
        gt_label = [last_element]
        for element in split_gt:
            if (element != last_element) and (element != 255):
                gt_label.append(element)
                last_element = element

        gt_split = torch.stack(gt_label).to(device)
        if (gt_split == 255).any().item() == False:
            gts_split.append(gt_split)

    return gts_split


def split_gt_feature(gt, feature, device, num_segments=16):

    N, T = gt.size()

    segment_length = T // num_segments

    gts_split = []
    features_split = []

    for i in range(N):
        for j in range(num_segments - 1):
            split_gt = gt[i, j * segment_length:(j + 1) * segment_length]
            gt_split = []
            last_element = split_gt[0]
            gt_label = [last_element]
            for element in split_gt:
                if (element != last_element) and (element != 255):
                    gt_label.append(element)
                    last_element = element

            gt_split = torch.stack(gt_label).to(device)
            if (gt_split == 255).any().item() == False:
                gts_split.append(gt_split)
                split_feature = feature[i, :, j * segment_length:(j + 1) * segment_length].mean(dim=-1)
                features_split.append(split_feature)

        split_gt = gt[i, (num_segments - 1) * segment_length:-1]
        gt_split = []
        last_element = split_gt[0]
        gt_label = [last_element]
        for element in split_gt:
            if (element != last_element) and (element != 255):
                gt_label.append(element)
                last_element = element

        gt_split = torch.stack(gt_label).to(device)
        if (gt_split == 255).any().item() == False:
            gts_split.append(gt_split)
            features_split.append(feature[i, :, (num_segments - 1) * segment_length:-1].mean(dim=-1))

    features_split = torch.stack(features_split).to(device)

    return gts_split, features_split

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num)) #（N，N）
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def gen_label_split(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num)) #（N，N）
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt


def generate_segment_features(output_feature, t_segment, device):
    segment_features = []

    for i in range(len(t_segment)):
        segment_list = t_segment[i]

        for j in range(len(segment_list)):
            start_frame = segment_list[j][2]
            end_frame = segment_list[j][3] + 1

            segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=-1)

            segment_features.append(segment_feature)

    segment_features = torch.stack(segment_features).to(device)

    return segment_features

def generate_freq_features(output_feature, t_segment, device, dataset):
    segment_features_allbatch = []

    for i in range(len(t_segment)):
        segment_list = t_segment[i]
        if len(segment_list) != 0:
            segment_features = []

            for j in range(len(segment_list)):
                start_frame = segment_list[j][2]
                end_frame = segment_list[j][3] + 1

                segment_feature = output_feature[i, :, start_frame:end_frame,:]

                _, t, _ = segment_feature.shape

                segment_feature = segment_feature.transpose(1,2)

                segment_feature = torch.fft.rfft(segment_feature, dim=2, norm="forward")

                segment_feature = torch.abs(segment_feature)

                segment_feature = F.interpolate(segment_feature, size=32, mode="linear")

                if dataset == "LARA":
                    l2norm = segment_feature.norm(dim=-1, keepdim=True)
                    l2norm[l2norm == 0] = 1
                    segment_feature = (segment_feature / l2norm) * 0.5

                segment_feature= segment_feature.flatten(0)

                segment_features.append(segment_feature)

            segment_features = torch.stack(segment_features).to(device)

            segment_features_allbatch.append(segment_features)

    return segment_features_allbatch

def generate_split_features(output_feature, start_ends, device):
    segment_features = []

    for i in range(len(start_ends)):
        segment_list = start_ends[i]

        for j in range(len(segment_list)):
            start_frame = segment_list[j][0]
            end_frame = segment_list[j][1] + 1

            segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=-1)

            segment_features.append(segment_feature)

    segment_features = torch.stack(segment_features).to(device)

    return segment_features

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True) #norm
    x2 = x2 / x2.norm(dim=-1, keepdim=True) #norm

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t() #sim
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def create_logits_freq(x1, logit_scale):

    # cosine similarity as logits
    x1_features_differences = x1.unsqueeze(1) - x1.unsqueeze(0)

    x1_features_distance = torch.norm(x1_features_differences, p=1, dim=-1)

    # shape = [global_batch_size, global_batch_size]
    return x1_features_distance