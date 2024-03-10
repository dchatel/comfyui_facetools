import os
import torch
import torchvision as tv
import numpy as np
import cv2
from insightface.utils import face_align
import mediapipe as mp
from scipy.spatial import ConvexHull
from folder_paths import models_dir
from .BiSeNet import BiSeNet
from ultralytics import YOLO
from onnxruntime import InferenceSession
from collections import namedtuple

def resize(img, size):
    h, w, _ = img.shape
    s = max(h, w)
    scale_factor = s / size
    ph, pw = (s - h) // 2, (s - w) // 2
    pad = tv.transforms.Pad((pw, ph))
    resize = tv.transforms.Resize(size=(size, size), antialias=True)
    img = resize(pad(img.permute(2,0,1))).permute(1,2,0)
    return img, scale_factor, ph, pw

class Models:
    @classmethod
    def yolo(cls, img, threshold):
        if '_yolo' not in cls.__dict__:
            cls._yolo = YOLO(os.path.join(models_dir,'ultralytics','bbox','face_yolov8m.pt'))
        dets = cls._yolo(img, conf=threshold)[0]
        return dets
    @classmethod
    def lmk(cls, crop):
        if '_lmk' not in cls.__dict__:
            cls._lmk = InferenceSession(os.path.join(models_dir, 'landmarks', 'fan2_68_landmark.onnx'))
        lmk = cls._lmk.run(None, {'input': crop})[0]
        return lmk

def get_submatrix_with_padding(img, a, b, c, d):
    pl = -min(a, 0)
    pt = -min(b, 0)
    pr = -min(img.shape[1] - c, 0)
    pb = -min(img.shape[0] - d, 0)
    a, b, c, d = max(a, 0), max(b, 0), min(c, img.shape[1]), min(d, img.shape[0])

    submatrix = img[b:d, a:c].permute(2,0,1)
    pad = tv.transforms.Pad((pl, pt, pr, pb))
    submatrix = pad(submatrix).permute(1,2,0)
    
    return submatrix

class Face:
    def __init__(self, img, a, b, c, d) -> None:
        self.img = img
        lmk = None
        best_score = 0
        i = 0
        crop = get_submatrix_with_padding(self.img, a, b, c, d)
        for curr_i in range(4):
            rcrop, s, ph, pw = resize(crop.rot90(curr_i), 256)
            rcrop = (rcrop[None] / 255).permute(0,3,1,2).type(torch.float32).numpy()
            curr_lmk = Models.lmk(rcrop)
            score = np.mean(curr_lmk[0,:,2])
            if score > best_score:
                best_score = score
                lmk = curr_lmk
                i = curr_i
        
        self.bbox = (a,b,c,d)
        self.confidence = best_score
        
        self.kps = np.vstack([
            lmk[0,[37,38,40,41],:2].mean(axis=0),
            lmk[0,[43,44,46,47],:2].mean(axis=0),
            lmk[0,[30,48,54],:2]
        ]) * 4 * s

        self.T2 = np.array([[1, 0, -a], [0, 1, -b], [0, 0, 1]])
        rot = cv2.getRotationMatrix2D((128*s,128*s), 90*i, 1)
        self.R = np.vstack((rot, np.array((0,0,1))))
    
    def crop(self, size, crop_factor):
        S = np.array([[1/crop_factor, 0, 0], [0, 1/crop_factor, 0], [0, 0, 1]])
        M = face_align.estimate_norm(self.kps, size)
        N = M @ self.R @ self.T2
        cx, cy = np.array((size/2, size/2, 1)) @ cv2.invertAffineTransform(M @ self.R @ self.T2).T
        T3 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        T4 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        N = N @ T4 @ S @ T3
        crop = cv2.warpAffine(self.img.numpy(), N, (size, size))
        
        return N, crop

def detect_faces(img, threshold):
    dets = Models.yolo((img[None] / 255).permute(0,3,1,2), threshold)
    boxes = (dets.boxes.xyxy.reshape(-1,2,2)).reshape(-1,4)
    faces = []
    for (a,b,c,d), box in zip(boxes.type(torch.int).cpu().numpy(), dets.boxes):
        cx, cy = (a+c)/2, (b+d)/2
        r = np.sqrt((c-a)**2 + (d-b)**2) / 2
        
        a,b,c,d = [int(x) for x in (cx - r, cy - r, cx + r, cy + r)]        
        face = Face(img, a, b, c, d)

        faces.append(face)
    return faces

### DEPRECATED ###
def squareimg(img):
    h, w, _ = img.shape
    s = max(h, w)
    ph, pw = (s - h) // 2, (s - w) // 2
    img = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT)
    return (img, ph, pw)

def transform_back(points, rot, pw, ph):
    points = np.hstack((points, np.ones((len(points),1)))) @ rot.T
    points -= np.array((pw, ph))
    return points

def get_candidates(img, face_analysis):
    simg, ph, pw = squareimg(img)
    h, w, _ = img.shape
    s = max(h, w)
    candidates = []
    for i in range(4):
        im = np.rot90(simg, i)
        rot = cv2.getRotationMatrix2D((s/2, s/2), -90*i, 1)
        dets = face_analysis.get(im)
        for face in dets:
            face.kps = transform_back(face.kps, rot, pw, ph)
            face.landmark_2d_106 = transform_back(face.landmark_2d_106, rot, pw, ph)
            
            # original bbox and size
            a,b,c,d = transform_back(face.bbox.reshape((2,2)), rot, pw, ph).flatten()
            a,c = min(a,c),max(a,c)
            b,d = min(b,d),max(b,d)
            face.w, face.h = abs(c-a), abs(d-b)
            
            face.centroid = np.array(((a+c)/2, (b+d)/2))
            face.s = max(abs(a-c), abs(b-d))
            
            face.angle = i*90
        candidates += dets
    return candidates

def group_faces(candidates):
    groups = []
    for face in candidates:
        group = None
        for g in groups:
            if any([np.linalg.norm(face.centroid - other.centroid) < np.sqrt(face.s)*2 for other in g]):
                group = g
                break
        if group is None:
            groups.append([face])
        else:
            group.append(face)
    return groups

def get_best_faces(groups):
    faces = [min(group, key=lambda face:abs(face.pose[2])) for group in groups]
    return faces

def get_faces(img, face_analysis):
    candidates = get_candidates(img, face_analysis)
    groups = group_faces(candidates)
    faces = get_best_faces(groups)
    return faces

def crop_faces(img, face, crop_size, crop_factor):
    M = face_align.estimate_norm(face.kps, crop_size, mode=None)
    cx, cy = face.kps.mean(axis=0)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    S = np.array([[1/crop_factor, 0, 0], [0, 1/crop_factor, 0], [0, 0, 1]])
    M = M @ T2 @ S @ T1
    crop = cv2.warpAffine(img, M, (crop_size,crop_size))
    
    return M, crop
### END_DEPRECATED ###

def get_face_mesh(crop):
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=10) as face_mesh:
        mesh = face_mesh.process(crop)
    h, w, _ = crop.shape
    if mesh.multi_face_landmarks is not None:
        all_pts = np.array([np.array([(w*l.x, h*l.y) for l in lmks.landmark]) for lmks in mesh.multi_face_landmarks], dtype=np.int32)
        idx = np.argmin(np.abs(all_pts - np.array([w/2,h/2])).sum(axis=(1,2)))
        points = all_pts[idx]
        return points
    else:
        return None

def mask_simple_square(face, M, crop):
    # rotated bbox and size
    a,b,c,d = face.bbox
    rect = np.array([
        [a,b,1],
        [a,d,1],
        [c,b,1],
        [c,d,1],
    ]) @ M.T
    lx, ly = [int(x) for x in np.min(rect, axis=0)]
    hx, hy = [int(x) for x in np.max(rect, axis=0)]
    mask = np.zeros((512,512), dtype=np.float32)
    mask = cv2.rectangle(mask, (lx,ly), (hx,hy), 1, -1)
    return mask

def mask_convex_hull(face, M, crop):
    points = get_face_mesh(crop)
    if points is None: return mask_simple_square(face, M, crop)
    hull = ConvexHull(points)
    mask = np.zeros((512,512), dtype=np.int32)
    cv2.fillPoly(mask, [points[hull.vertices,:]], color=1)
    mask = mask.astype(np.float32)
    return mask

def mask_BiSeNet(face, M, crop):
    with torch.no_grad():
        bisenet = BiSeNet(n_classes=19)
        bisenet.cuda()
        model_path = os.path.join(models_dir, 'bisenet', '79999_iter.pth')
        bisenet.load_state_dict(torch.load(model_path))
        bisenet.eval()
        crop_t = torch.from_numpy(crop)[None].permute(0,3,1,2).contiguous().float().div(255).cuda()
        segms_t = bisenet(crop_t)[0].argmax(1).float()
    # convert segmentation to mask
    # 1: skin
    # 2: l_brow
    # 3: r_brow
    # 4: l_eye
    # 5: r_eye
    # 6: eye_g
    # 7: l_ear
    # 8: r_ear
    # 9: ear_r
    # 10: nose
    # 11: mouth
    # 12: u_lip
    # 13: l_lip
    # 14: neck
    # 15: neck_l
    # 16: cloth
    # 17: hair
    # 18: hat
    face_part_ids = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13]).cuda()
    segms_t = torch.sum(segms_t.repeat(len(face_part_ids), 1,1,1) == face_part_ids[...,None,None,None], axis=0).float()
    mask = segms_t[0].cpu()
    return mask

mask_types = [
    'simple_square',
    'convex_hull',
    'BiSeNet',
]

mask_funs = {
    'simple_square': mask_simple_square,
    'convex_hull': mask_convex_hull,
    'BiSeNet': mask_BiSeNet,
}

def mask_crop(face, M, crop, mask_type):
    mask = mask_funs[mask_type](face, M, crop)
    return mask
