"""Cheap image-plane consistency checks; these are not a 6D accuracy certificate."""
import numpy as np


def projected_bbox(bounds,pose,K,shape):
    bounds=np.asarray(bounds)
    corners=np.array([[bounds[x,0],bounds[y,1],bounds[z,2]] for x in (0,1) for y in (0,1) for z in (0,1)])
    camera=corners@pose[:3,:3].T+pose[:3,3]
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=.001):return None
    pixels=camera@K.T;pixels=pixels[:,:2]/pixels[:,2,None]
    height,width=shape[:2]
    low=np.clip(pixels.min(axis=0),[0,0],[width,height]);high=np.clip(pixels.max(axis=0),[0,0],[width,height])
    if np.any(high-low<1):return None
    return [*low.tolist(),*high.tolist()]


def bbox_agreement(predicted,detected):
    if predicted is None or detected is None:return None
    a=np.asarray(predicted,dtype=float);b=np.asarray(detected,dtype=float)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():return None
    sa=a[2:]-a[:2];sb=b[2:]-b[:2]
    if np.any(sa<=0) or np.any(sb<=0):return None
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    iou=float(intersection/(sa.prod()+sb.prod()-intersection))
    offset=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/np.linalg.norm(sb))
    return {'iou':iou,'center_offset':offset,'consistent':iou>=.25 and offset<=.5}
