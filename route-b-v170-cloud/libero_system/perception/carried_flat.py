"""Register a current flat body using depth, measured jaw plane and public solids."""
from dataclasses import replace
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from .carried_mug import _measure, _align
from .flat_mesh import pudding_triangles, public_array, PHYSICAL_EXTENTS, PHYSICAL_CENTER_FROM_VISUAL
from dataclasses import dataclass

@dataclass(frozen=True)
class CarriedFlatFit:
    center_world: np.ndarray
    axes_world: np.ndarray
    extents_m: np.ndarray
    matched_pixels: int
    free_space_conflicts: int
    rendered_pixels: int
from libero_system.perception.mesh_depth import render_depth
from libero_system.perception.geometry import backproject_frame


def public_hand_penetration_test(robot,half):
    width=robot.gripper_width_m
    palm=public_array('PALM');finger=public_array('FINGER')
    # Interior payload samples and a 2mm penetration tolerance avoid rejecting contact.
    volume=np.array(list(product(np.linspace(-.9,.9,5),repeat=3)))*np.maximum(half-.002,0.)
    def intersects(rotation,center):
        local=(volume@rotation.T+center-robot.ee_pose.position)@robot.ee_pose.rotation
        for points,eq in ((local+[0,0,.097],palm),
                          (local-[0,width/2,.0524-.097],finger),
                          ((local-[0,-width/2,.0524-.097])@np.diag([-1.,-1.,1.]),finger)):
            if np.any(np.all(points@eq[:,:3].T+eq[:,3]<-.002,axis=1)):return True
        return False
    return intersects

def fit_carried_flat_package(frames, source, robot):
    if (source is None or source.name!='chocolate pudding' or not frames
            or source.surface_points_world is None or len(source.surface_points_world)<100):
        raise LookupError('no current identified flat-package surface')
    if not .90*PHYSICAL_EXTENTS[1]<=robot.gripper_width_m<=1.08*PHYSICAL_EXTENTS[1]:
        raise LookupError('measured aperture does not establish a full-width flat pinch')
    triangles=pudding_triangles()
    half=np.ptp(triangles.reshape(-1,3),axis=0)/2
    intersects=public_hand_penetration_test(robot,half)
    verts=np.array(list(product((-1.,1.),repeat=3)))*half
    faces=[]
    for axis in range(3):
        for side in (-1.,1.):
            ids=np.flatnonzero(np.isclose(verts[:,axis],side*half[axis]))
            a,b,c,d=ids;faces.extend(((a,b,c),(b,c,d)))
    box=verts[np.array(faces)]
    small=[]
    for frame in frames:
        intr=frame.intrinsics.copy();intr[:2]/=2
        if frame.observation_v_flipped:intr[1,2]-=.5
        small.append(replace(frame,rgb=frame.rgb[::2,::2],depth_m=frame.depth_m[::2,::2],intrinsics=intr))
    options=[]
    for angle in range(0,180,15):
        rot=robot.ee_pose.rotation@Rotation.from_euler('y',angle,degrees=True).as_matrix()
        for x,z in product((-.04,-.02,0.,.02,.04),repeat=2):
            center=robot.ee_pose.position+robot.ee_pose.rotation@np.array([x,0.,z])
            if np.linalg.norm(center-source.centroid_world)>.08:continue
            if intersects(rot,center):continue
            metrics=_measure(box,rot,center,small)
            options.append((metrics,rot,center))
    options.sort(key=lambda x:x[0][0],reverse=True)
    refinements=[]
    for _,rot,center in options[:5]:
        for x,z in product((-.01,0.,.01),repeat=2):
            pos=center+robot.ee_pose.rotation@np.array([x,0.,z])
            if intersects(rot,pos):continue
            metrics=_measure(box,rot,pos,small)
            refinements.append((metrics,rot,pos))
    refinements.sort(key=lambda x:x[0][0],reverse=True)
    best=None
    for _,rotation,start in refinements[:5]:
        center=start.copy();rot=rotation.copy()
        for _ in range(15):
            models=[];observed=[]
            for frame in frames:
                predicted=render_depth(triangles@rot.T+center,frame)
                mask=np.isfinite(predicted)&np.isfinite(frame.depth_m)&(frame.depth_m>.05)&(abs(predicted-frame.depth_m)<.025)
                models.append(backproject_frame(replace(frame,depth_m=predicted),mask=mask,stride=2).points_world)
                observed.append(backproject_frame(frame,mask=mask,stride=2).points_world)
            model=np.concatenate(models);actual=np.concatenate(observed)
            if len(model)<100:break
            delta,shift=_align(model,actual);new_center=delta@center+shift
            if np.linalg.norm(new_center-source.centroid_world)>.08:break
            new_rot=delta@rot
            if intersects(new_rot,new_center):break
            center=new_center;rot=new_rot
            metrics=_measure(triangles,rot,center,frames)
            if best is None or metrics[0]>best[0][0]:best=(metrics,rot.copy(),center.copy())
    if best is None:raise LookupError('no supported body refinement')
    metrics,rot,center=best;_,matched,conflicts,visible=metrics
    if matched<500 or matched<.50*visible or conflicts>.08*visible:raise LookupError('flat mesh lacks consistent visible depth support')
    return CarriedFlatFit(center+rot@PHYSICAL_CENTER_FROM_VISUAL,rot,PHYSICAL_EXTENTS.copy(),matched,conflicts,visible)
