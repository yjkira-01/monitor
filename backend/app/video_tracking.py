"""Short-lived optical-flow boxes between asynchronous person detections."""
import cv2
import numpy as np


def tracking_gray(frame):
    height, width = frame.shape[:2]
    return cv2.cvtColor(cv2.resize(frame, (320, max(1, round(height*320/width)))), cv2.COLOR_BGR2GRAY)


class PersonBoxTracker:
    def __init__(self):
        self.reference_time = -1.
        self.previous = None
        self.targets = []

    def update(self, frame, reference, position):
        gray = tracking_gray(frame)
        if reference is None or position-reference[1] > 1.5 or position < reference[1]:
            self.targets = []
            self.previous = gray
            return []
        source, timestamp, people = reference
        height, width = gray.shape
        if timestamp != self.reference_time:
            self.reference_time = timestamp
            self.previous = source
            self.targets = []
            for person in people:
                bbox = np.asarray(person['bbox'], dtype=np.float32)
                if bbox.shape != (4,) or not np.isfinite(bbox).all():
                    continue
                x1, y1, x2, y2 = (bbox*[width, height, width, height]).astype(int)
                mask = np.zeros_like(source)
                dx, dy = (x2-x1)//5, (y2-y1)//5
                mask[max(0,y1+dy):min(height,y2-dy), max(0,x1+dx):min(width,x2-dx)] = 255
                points = cv2.goodFeaturesToTrack(source, maxCorners=32, qualityLevel=.02, minDistance=4, mask=mask)
                if points is not None and len(points) >= 4:
                    self.targets.append((person, bbox, points))
        if self.previous.shape != gray.shape:
            self.targets = []
        tracked = []
        for person, bbox, points in self.targets:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None, winSize=(21,21), maxLevel=3)
            if moved is None:
                continue
            back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, winSize=(21,21), maxLevel=3)
            if back is None:
                continue
            keep = status.ravel().astype(bool) & reverse.ravel().astype(bool) & np.isfinite(moved).all((1,2))
            keep &= np.linalg.norm(back-points, axis=2).ravel() < 1.5
            if keep.sum() < 4:
                continue
            delta = np.median((moved-points)[keep, 0], axis=0)/[width,height]
            if np.linalg.norm(delta) > .3:
                continue
            bbox = np.clip(bbox+np.tile(delta,2),0,1)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            tracked.append((person, bbox, moved[keep]))
        self.targets, self.previous = tracked, gray
        return [{**person, 'bbox': bbox.tolist()} for person, bbox, _ in tracked]
