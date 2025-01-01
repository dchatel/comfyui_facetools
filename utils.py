import os
import torch
import torchvision as tv
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import ConvexHull
from folder_paths import models_dir
from .BiSeNet import BiSeNet
from ultralytics import YOLO
from onnxruntime import InferenceSession, get_available_providers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def pad_to_stride(image, stride=32):
    h, w, _ = image.shape
    pr = (stride - w % stride) % stride
    pb = (stride - h % stride) % stride
    padded_image = tv.transforms.transforms.F.pad(image.permute(2,0,1), (0, 0, pr, pb)).permute(1,2,0)
    return padded_image

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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._yolo = cls._yolo.to(device)
        dets = cls._yolo(img, conf=threshold)[0]
        return dets
    @classmethod
    def lmk(cls, crop):
        if '_lmk' not in cls.__dict__:
            cls._lmk = InferenceSession(os.path.join(models_dir, 'landmarks', 'fan2_68_landmark.onnx'), providers=get_available_providers())
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
        self.w = c - a
        self.h = d - b
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
        M = estimate_norm(self.kps, size)
        N = M @ self.R @ self.T2
        cx, cy = np.array((size/2, size/2, 1)) @ cv2.invertAffineTransform(M @ self.R @ self.T2).T
        T3 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        T4 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        N = N @ T4 @ S @ T3
        crop = cv2.warpAffine(self.img.numpy(), N, (size, size))
        crop = torch.from_numpy(crop)[None]
        
        return N, crop

def detect_faces(img, threshold):
    img = pad_to_stride(img, stride=32)
    dets = Models.yolo((img[None] / 255).permute(0,3,1,2), threshold)
    boxes = (dets.boxes.xyxy.reshape(-1,2,2)).reshape(-1,4)
    faces = []
    for (a,b,c,d), box in zip(boxes.type(torch.int).cpu().numpy(), dets.boxes):
        cx, cy = (a+c)/2, (b+d)/2
        r = np.sqrt((c-a)**2 + (d-b)**2) / 2
        
        a,b,c,d = [int(x) for x in (cx - r, cy - r, cx + r, cy + r)]        
        face = Face(img, a, b, c, d)
        
        M = estimate_norm(face.kps, 512)
        if abs(M[0,0]) > 1 and abs(M[1,1]) > 1:
            faces.append(face)
    return faces

def get_face_mesh(crop: torch.Tensor):
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=10) as face_mesh:
        mesh = face_mesh.process(crop.mul(255).type(torch.uint8)[0].numpy())
    _, h, w, _ = crop.shape
    if mesh.multi_face_landmarks is not None:
        all_pts = np.array([np.array([(w*l.x, h*l.y) for l in lmks.landmark]) for lmks in mesh.multi_face_landmarks], dtype=np.int32)
        idx = np.argmin(np.abs(all_pts - np.array([w/2,h/2])).sum(axis=(1,2)))
        points = all_pts[idx]
        return points
    else:
        return None

def mask_simple_square(face, M, crop):
    # rotated bbox and size
    h,w = crop.shape[1:3]
    a,b,c,d = face.bbox
    rect = np.array([
        [a,b,1],
        [a,d,1],
        [c,b,1],
        [c,d,1],
    ]) @ M.T
    lx, ly = [int(x) for x in np.min(rect, axis=0)]
    hx, hy = [int(x) for x in np.max(rect, axis=0)]
    mask = np.zeros((h,w), dtype=np.float32)
    mask = cv2.rectangle(mask, (lx,ly), (hx,hy), 1, -1)
    mask = torch.from_numpy(mask)[None]
    return mask

def mask_convex_hull(face, M, crop):
    h,w = crop.shape[1:3]
    points = get_face_mesh(crop)
    if points is None: return mask_simple_square(face, M, crop)
    hull = ConvexHull(points)
    mask = np.zeros((h,w), dtype=np.int32)
    cv2.fillPoly(mask, [points[hull.vertices,:]], color=1)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask[None])
    return mask

def mask_BiSeNet(crop,
                 skin=True,
                 l_brow=True,
                 r_brow=True,
                 l_eye=True,
                 r_eye=True,
                 eye_g=True,
                 l_ear=True,
                 r_ear=True,
                 ear_r=True,
                 nose=True,
                 mouth=True,
                 u_lip=True,
                 l_lip=True,
                 neck=False,
                 neck_l=False,
                 cloth=False,
                 hair=False,
                 hat=False,
                 ):
    with torch.no_grad():
        bisenet = BiSeNet(n_classes=19)
        bisenet.cuda()
        model_path = os.path.join(models_dir, 'bisenet', '79999_iter.pth')
        bisenet.load_state_dict(torch.load(model_path))
        bisenet.eval()
        crop_t = crop.permute(0,3,1,2).cuda().float()
        segms_t = bisenet(crop_t)[0].argmax(1).float()
        
    dic = {
        'skin': 1,
        'l_brow': 2,
        'r_brow': 3,
        'l_eye': 4,
        'r_eye': 5,
        'eye_g': 6,
        'l_ear': 7,
        'r_ear': 8,
        'ear_r': 9,
        'nose': 10,
        'mouth': 11,
        'u_lip': 12,
        'l_lip': 13,
        'neck': 14,
        'neck_l': 15,
        'cloth': 16,
        'hair': 17,
        'hat': 18,
    }
    keep = []
    for k, v in locals().items():
        if k in dic and v:
            keep.append(dic[k])

    face_part_ids = torch.tensor(keep).cuda()
    segms_t = torch.sum(segms_t.repeat(len(face_part_ids), 1,1,1) == face_part_ids[...,None,None,None], axis=0).float()
    mask = segms_t.cpu()
    return mask

def mask_jonathandinu(crop, skin=True, nose=True, eye_g=True, l_eye=True, r_eye=True, l_brow=True, r_brow=True,
                    l_ear=True, r_ear=True, mouth=True, u_lip=True, l_lip=True,
                    hair=False, hat=False, ear_r=False, neck_l=False, neck=False, cloth=False):
    global jonathandinu_image_processor, jonathandinu_model
 
    device = (
        "cuda"
        # Device for NVIDIA or AMD GPUs
        if torch.cuda.is_available()
        else "mps"
        # Device for Apple Silicon (Metal Performance Shaders)
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    if 'jonathandinu_image_processor' not in globals():
        jonathandinu_image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        jonathandinu_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        jonathandinu_model.to(device)

    inputs = jonathandinu_image_processor(images=crop.mul(255).type(torch.uint8), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = jonathandinu_model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = tv.transforms.functional.resize(logits, crop.shape[1:3], antialias=True)
        
    labels = upsampled_logits.argmax(dim=1)
    
    ids = {
        'skin': 1,
        'nose': 2,
        'eye_g': 3,
        'l_eye': 4,
        'r_eye': 5,
        'l_brow': 6,
        'r_brow': 7,
        'l_ear': 8,
        'r_ear': 9,
        'mouth': 10,
        'u_lip': 11,
        'l_lip': 12,
        'hair': 13,
        'hat': 14,
        'ear_r': 15,
        'neck_l': 16,
        'neck': 17,
        'cloth': 18,
    }
    keep = []
    for k, v in locals().items():
        if k in ids and v:
            keep.append(ids[k])
    face_part_ids = torch.tensor(keep).cuda()

    mask = torch.sum(labels.repeat(len(face_part_ids), 1,1,1) == face_part_ids[...,None,None,None], axis=0).float().cpu()

    return mask

mask_types = [
    'simple_square',
    'convex_hull',
    'BiSeNet',
    'jonathandinu',
    # 'clean BiSeNet',
]

mask_funs = {
    'simple_square': mask_simple_square,
    'convex_hull': mask_convex_hull,
    'BiSeNet': lambda face, M, crop: mask_BiSeNet(crop),
    'jonathandinu': lambda face, M, crop: mask_jonathandinu(crop),
    # 'clean BiSeNet': mask_clean_BiSeNet,
}

def mask_crop(face, M, crop, mask_type):
    mask = mask_funs[mask_type](face, M, crop)
    return mask
