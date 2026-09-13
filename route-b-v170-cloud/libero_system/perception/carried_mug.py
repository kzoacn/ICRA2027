"""Fit a known public cup surface to its current visible RGB-D observations."""
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import backproject_frame
from .mesh_depth import render_depth
from .mug_mesh import yellow_white_mug_triangles


@dataclass(frozen=True)
class CarriedMugFit:
    center_world: np.ndarray
    axes_world: np.ndarray
    extents_m: np.ndarray
    matched_pixels: int
    free_space_conflicts: int
    rendered_pixels: int


@lru_cache(maxsize=1)
def _template():
    triangles=yellow_white_mug_triangles()
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    rng=np.random.default_rng(0);indices=rng.choice(len(triangles),18000,p=area/area.sum())
    uv=rng.random((len(indices),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    samples=(triangles[indices,0]+uv[:,:1]*(triangles[indices,1]-triangles[indices,0])
             +uv[:,1:]*(triangles[indices,2]-triangles[indices,0]))
    return triangles,samples,cKDTree(samples)


def _align(model,observed):
    m=model.mean(0);o=observed.mean(0)
    u,_,vt=np.linalg.svd((model-m).T@(observed-o))
    rotation=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
    return rotation,o-rotation@m


def _measure(triangles,rotation,center,frames):
    matched=conflicts=visible=0
    for frame in frames:
        predicted=render_depth(triangles@rotation.T+center,frame)
        valid=np.isfinite(predicted)&np.isfinite(frame.depth_m)&(frame.depth_m>.05)
        delta=predicted[valid]-frame.depth_m[valid]
        visible+=int(valid.sum());matched+=int((abs(delta)<.006).sum())
        conflicts+=int((delta<-.008).sum())
    return matched-3*conflicts,matched,conflicts,visible


def fit_carried_yellow_white_mug(frames,source):
    """Return a current fitted pose, or decline when the surface is ambiguous.

    The crop supplies identity and a bounded search neighbourhood. Surface
    correspondences propose rigid poses; perspective depth from both cameras
    rejects hypotheses that occupy measured free space. Occluding fingers are
    allowed in front of the predicted cup. No initial grasp rigid offset is
    assumed.
    """
    if source is None or source.name!='yellow and white mug' or source.surface_points_world is None or not frames:
        raise LookupError('no current identified cup surface')
    points=np.asarray(source.surface_points_world,dtype=float)
    points=points[np.all(np.isfinite(points),axis=1)]
    if len(points)<100:raise LookupError('too few current cup points')
    if len(points)>3000:points=points[np.linspace(0,len(points)-1,3000,dtype=int)]
    triangles,samples,tree=_template();seed_center=source.centroid_world;options=[]
    for yaw in (0.,180.):
        for tilt in ((0.,0.),(20.,0.),(-20.,0.),(0.,20.),(0.,-20.)):
            rotation=source.axes_world@Rotation.from_euler('zyx',[yaw,*tilt],degrees=True).as_matrix()
            center=seed_center.copy()
            for _ in range(18):
                distance,indices=tree.query((points-center)@rotation)
                keep=distance<max(.008,min(.025,float(np.quantile(distance,.65))))
                if keep.sum()<60:break
                proposal_rotation,proposal_center=_align(samples[indices[keep]],points[keep])
                if proposal_rotation[2,2]<.60 or np.linalg.norm(proposal_center-seed_center)>.08:break
                rotation,center=proposal_rotation,proposal_center
            metrics=_measure(triangles,rotation,center,frames)
            options.append((metrics,rotation.copy(),center.copy()))
    best=max(options,key=lambda item:item[0][0]);rotation=best[1].copy();center=best[2].copy()
    # Projective correspondences refine the visible mesh rather than the
    # occluded crop's bounding-box centre. Keep only matching depth layers.
    for _ in range(12):
        models=[];observed=[]
        for frame in frames:
            predicted=render_depth(triangles@rotation.T+center,frame)
            mask=np.isfinite(predicted)&np.isfinite(frame.depth_m)&(frame.depth_m>.05)
            mask &= abs(predicted-frame.depth_m)<.016
            models.append(backproject_frame(replace(frame,depth_m=predicted),mask=mask,stride=2).points_world)
            observed.append(backproject_frame(frame,mask=mask,stride=2).points_world)
        model=np.concatenate(models);actual=np.concatenate(observed)
        if len(model)<100:break
        delta,translation=_align(model,actual)
        next_center=delta@center+translation;next_rotation=delta@rotation
        if next_rotation[2,2]<.60 or np.linalg.norm(next_center-seed_center)>.08:break
        center,rotation=next_center,next_rotation
        metrics=_measure(triangles,rotation,center,frames)
        if metrics[0]>best[0][0]:best=(metrics,rotation.copy(),center.copy())
    metrics,rotation,center=best;_,matched,conflicts,visible=metrics
    if matched<500 or matched<.50*visible or conflicts>.08*visible:
        raise LookupError('cup mesh lacks consistent visible depth support')
    extents=np.ptp(triangles.reshape(-1,3),axis=0)
    return CarriedMugFit(center,rotation,extents,matched,conflicts,visible)
