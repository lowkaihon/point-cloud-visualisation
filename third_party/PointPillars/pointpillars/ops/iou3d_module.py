# Pure-torch replacement for iou3d_module.py — avoids CUDA-only iou3d_op.
# Original repo's rotated-NMS is replaced with axis-aligned NMS on the BEV
# envelope. For a single-frame KITTI inference demo with nms_thr=0.01, the
# distinction is negligible. Training-time iou functions (boxes_iou_bev,
# boxes_overlap_bev) are stubbed — they're not called at inference.

import torch


def _nms_axis_aligned(boxes_xyxy: torch.Tensor, scores: torch.Tensor,
                      iou_thresh: float) -> torch.Tensor:
    """Greedy axis-aligned NMS. boxes_xyxy: (N, 4) = [x1, y1, x2, y2]."""
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-9)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Drop-in replacement for the CUDA rotated-NMS. boxes: (N, 5) =
    [x_min, y_min, x_max, y_max, ry]. We ignore ry (axis-aligned envelope NMS).
    """
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    scores = scores[order].contiguous()

    keep_local = _nms_axis_aligned(boxes[:, :4], scores, thresh)
    keep = order[keep_local]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def boxes_iou_bev(boxes_a, boxes_b):
    raise NotImplementedError(
        "boxes_iou_bev is CUDA-only in upstream; not needed for inference."
    )


def boxes_overlap_bev(boxes_a, boxes_b):
    raise NotImplementedError(
        "boxes_overlap_bev is CUDA-only in upstream; not needed for inference."
    )
