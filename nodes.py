import torch
from collections import defaultdict
from .utils import *
from transformers import pipeline

class GenderFaceFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'faces': ('FACE',),
                'gender': (['man', 'woman'],)
            }
        }
    
    RETURN_TYPES = ('FACE', 'FACE')
    RETURN_NAMES = ('filtered', 'rest')
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    def run(self, faces, gender):
        filtered = []
        rest = []
        pipe = pipeline('image-classification', model='dima806/man_woman_face_image_detection', device=0)
        for face in faces:
            _, im = face.crop(224, 1.2)
            im = im.permute(0,3,1,2)[0]
            im = tv.transforms.functional.resize(im, (224,224))
            r = pipe(tv.transforms.functional.to_pil_image(im))
            idx = np.argmax([i['score'] for i in r])
            if r[idx]['label'] == gender:
                filtered.append(face)
            else:
                rest.append(face)
        return (filtered, rest)

class OrderedFaceFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'faces': ('FACE',),
                'criteria': (['area'],),
                'order': (['descending', 'ascending'],),
                'take_start': ('INT', {'default': 0, 'min': 0, 'step': 1}),
                'take_count': ('INT', {'default': 1, 'min': 1, 'step': 1}),
            }
        }
    
    RETURN_TYPES = ('FACE', 'FACE')
    RETURN_NAMES = ('filtered', 'rest')
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    def run(self, faces, criteria, order, take_start, take_count):
        filtered = []
        rest = []
        funs = {
            'area': lambda face: face.w * face.h
        }
        sorted_faces = sorted(faces, key=funs[criteria], reverse=order == 'descending')
        filtered = sorted_faces[take_start:take_start+take_count]
        rest = sorted_faces[:take_start] + sorted_faces[take_start+take_count:]
        return (filtered, rest)

class DetectFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'min_size': ('INT', {'default': 64, 'max': 512, 'step': 8}),
                'max_size': ('INT', {'default': 512, 'min': 512, 'step': 8}),
            },
            'optional': {
                'mask': ('MASK',),
            }
        }
    
    RETURN_TYPES = ('FACE',)
    RETURN_NAMES = ('faces',)
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    def run(self, image, threshold, min_size, max_size, mask=None):
        faces = []
        masked = image
        if mask is not None:
            masked = image * tv.transforms.functional.resize(1-mask, image.shape[1:3])[..., None]
        masked = (masked * 255).type(torch.uint8)
        for i, img in enumerate(masked):
            unfiltered_faces = detect_faces(img, threshold)
            for face in unfiltered_faces:
                a, b, c, d = face.bbox
                h = abs(d-b)
                w = abs(c-a)
                if (h <= max_size or w <= max_size) and (min_size <= h or min_size <= w):
                    face.image_idx = i
                    face.img = image[i]
                    faces.append(face)
        return (faces,)

class CropFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'faces': ('FACE',),
                'crop_size': ('INT', {'default': 512, 'min': 512, 'max': 1024, 'step': 128}),
                'crop_factor': ('FLOAT', {'default': 1.5, 'min': 1.0, 'max': 3, 'step': 0.1}),
                'mask_type': (mask_types,)
            }
        }
    
    RETURN_TYPES = ('IMAGE', 'MASK', 'WARP')
    RETURN_NAMES = ('crops', 'masks', 'warps')
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    def run(self, faces, crop_size, crop_factor, mask_type):
        if len(faces) == 0:
            empty_crop = torch.zeros((1,512,512,3))
            empty_mask = torch.zeros((1,512,512))
            empty_warp = np.array([
                [1,0,-512],
                [0,1,-512],
            ], dtype=np.float32)
            return (empty_crop, empty_mask, [empty_warp])
        
        crops = []
        masks = []
        warps = []
        for face in faces:
            M, crop = face.crop(crop_size, crop_factor)
            mask = mask_crop(face, M, crop, mask_type)
            crops.append(np.array(crop[0]))
            masks.append(np.array(mask[0]))
            warps.append(M)
        crops = torch.from_numpy(np.array(crops)).type(torch.float32)
        masks = torch.from_numpy(np.array(masks)).type(torch.float32)
        return (crops, masks, warps)
    
class WarpFaceBack:
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE',),
                'face': ('FACE',),
                'crop': ('IMAGE',),
                'mask': ('MASK',),
                'warp': ('WARP',),
            }
        }
    
    def run(self, images, face, crop, mask, warp):
        groups = defaultdict(list)
        for f,c,m,w in zip(face, crop, mask, warp):
            groups[f.image_idx].append((f.img,c,m,w))

        results = []
        for i, image in enumerate(images):
            if i not in groups:
                result = image
            else:
                values = groups[i]
                crop, mask, warp = list(zip(*[x[1:] for x in values]))
                warped_masks = [np.clip(cv2.warpAffine(single_mask.numpy(),
                                cv2.invertAffineTransform(single_warp),
                                image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4), 0, 1)
                                for single_warp, single_mask in zip(warp, mask)]
                full_mask = np.add.reduce(warped_masks, axis=0)[...,None]
                swapped = np.add.reduce([
                    np.clip(cv2.warpAffine(single_crop.cpu().numpy(),
                                cv2.invertAffineTransform(single_warp),
                                image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4
                                ), 0, 1) * single_mask[..., None]
                    for single_crop, single_mask, single_warp in zip(crop, warped_masks, warp)
                ], axis=0) / np.maximum(1, full_mask)
                full_mask = np.minimum(1, full_mask)
                result = (swapped + (1 - full_mask) * image.numpy())
                result = torch.from_numpy(result)
            results.append(result)

        results = torch.stack(results)
        return (results, )
    
class MergeWarps:
    RETURN_TYPES = ('IMAGE','MASK','WARP')
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'crop0': ('IMAGE',),
                'mask0': ('MASK',),
                'warp0': ('WARP',),
                'crop1': ('IMAGE',),
                'mask1': ('MASK',),
                'warp1': ('WARP',),
            }
        }
    
    def run(self, crop0, mask0, warp0, crop1, mask1, warp1):
        crops = torch.vstack((crop0, crop1))
        masks = torch.vstack((mask0, mask1))
        warps = warp0 + warp1
        return (crops, masks, warps)

class BiSeNetMask:
    RETURN_TYPES = ('MASK',)
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'crop': ('IMAGE',),
                'skin': ('BOOLEAN', {'default': True}),
                'left_brow': ('BOOLEAN', {'default': True}),
                'right_brow': ('BOOLEAN', {'default': True}),
                'left_eye': ('BOOLEAN', {'default': True}),
                'right_eye': ('BOOLEAN', {'default': True}),
                'eyeglasses': ('BOOLEAN', {'default': True}),
                'left_ear': ('BOOLEAN', {'default': True}),
                'right_ear': ('BOOLEAN', {'default': True}),
                'earring': ('BOOLEAN', {'default': True}),
                'nose': ('BOOLEAN', {'default': True}),
                'mouth': ('BOOLEAN', {'default': True}),
                'upper_lip': ('BOOLEAN', {'default': True}),
                'lower_lip': ('BOOLEAN', {'default': True}),
                'neck': ('BOOLEAN', {'default': False}),
                'necklace': ('BOOLEAN', {'default': False}),
                'cloth': ('BOOLEAN', {'default': False}),
                'hair': ('BOOLEAN', {'default': False}),
                'hat': ('BOOLEAN', {'default': False}),
            }
        }
    
    def run(self, crop, skin, left_brow, right_brow, left_eye, right_eye, eyeglasses,
            left_ear, right_ear, earring, nose, mouth, upper_lip, lower_lip,
            neck, necklace, cloth, hair, hat):
        masks = mask_BiSeNet(crop, skin, left_brow, right_brow, left_eye, right_eye, eyeglasses,
            left_ear, right_ear, earring, nose, mouth, upper_lip, lower_lip,
            neck, necklace, cloth, hair, hat)
        return (masks, )

class JonathandinuMask:
    RETURN_TYPES = ('MASK',)
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'crop': ('IMAGE',),
                'skin': ('BOOLEAN', {'default': True}),
                'nose': ('BOOLEAN', {'default': True}),
                'eyeglasses': ('BOOLEAN', {'default': False}),
                'left_eye': ('BOOLEAN', {'default': True}),
                'right_eye': ('BOOLEAN', {'default': True}),
                'left_brow': ('BOOLEAN', {'default': True}),
                'right_brow': ('BOOLEAN', {'default': True}),
                'left_ear': ('BOOLEAN', {'default': True}),
                'right_ear': ('BOOLEAN', {'default': True}),
                'mouth': ('BOOLEAN', {'default': True}),
                'upper_lip': ('BOOLEAN', {'default': True}),
                'lower_lip': ('BOOLEAN', {'default': True}),
                'hair': ('BOOLEAN', {'default': False}),
                'hat': ('BOOLEAN', {'default': False}),
                'earring': ('BOOLEAN', {'default': False}),
                'necklace': ('BOOLEAN', {'default': False}),
                'neck': ('BOOLEAN', {'default': False}),
                'cloth': ('BOOLEAN', {'default': False}),
            }
        }
    
    def run(self, crop, skin, nose, eyeglasses, left_eye, right_eye, left_brow, right_brow, left_ear, right_ear,
            mouth, upper_lip, lower_lip, hair, hat, earring, necklace, neck, cloth):
        masks = mask_jonathandinu(crop, skin, nose, eyeglasses, left_eye, right_eye, left_brow, right_brow, left_ear, right_ear,
                             mouth, upper_lip, lower_lip, hair, hat, earring, necklace, neck, cloth)
        return (masks, )

NODE_CLASS_MAPPINGS = {
    'DetectFaces': DetectFaces,
    'CropFaces': CropFaces,
    'WarpFacesBack': WarpFaceBack,
    'BiSeNetMask': BiSeNetMask,
    'JonathandinuMask': JonathandinuMask,
    'MergeWarps': MergeWarps,
    'GenderFaceFilter': GenderFaceFilter,
    'OrderedFaceFilter': OrderedFaceFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'DetectFaces': 'DetectFaces',
    'CropFaces': 'CropFaces',
    'WarpFacesBack': 'Warp Faces Back',
    'BiSeNetMask': 'BiSeNet Mask',
    'JonathandinuMask': 'Jonathandinu Mask',
    'MergeWarps': 'Merge Warps',
    'GenderFaceFilter': 'Gender Face Filter',
    'OrderedFaceFilter': 'Ordered Face Filter',
}
