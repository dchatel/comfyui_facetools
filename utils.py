import os
import torch
import numpy as np
import cv2
from insightface.utils import face_align
import mediapipe as mp
from scipy.spatial import ConvexHull
from folder_paths import models_dir
from .BiSeNet import BiSeNet

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
    mask = np.zeros((512,512), dtype=np.int32)
    mask = cv2.rectangle(mask, (lx,ly), (hx,hy), 1, -1)
    mask = mask.astype(np.float32)
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
