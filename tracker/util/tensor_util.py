import torch.nn.functional as F


def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    return (intersection + 1e-6) / (union + 1e-6) 

# STM
def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]

    new_h = h + d - h % d if h % d > 0 else h
    new_w = w + d - w % d if w % d > 0 else w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = lw, int(uw), lh, int(uh)
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if len(img.shape) == 4:
        if pad[2]+pad[3] > 0:
            img = img[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,:,pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2]+pad[3] > 0:
            img = img[:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img