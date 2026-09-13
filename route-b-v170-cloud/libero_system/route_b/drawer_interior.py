"""Constrain a visible drawer support by its measured handle and physical walls."""
from types import SimpleNamespace

import numpy as np

from .models import SceneObject


def drawer_interior_from_points(points, handle, preferred, source, *, require_grounded_height=True):
    """Return a contained support patch centred near the existing placement.

    The public three-drawer cabinet collision boxes have inner front/back
    faces 36.78/188.82 mm behind the handle centre and 205 mm clear width.
    A 4 mm inset allows for surface/centre differences in the observed bar.
    RGB-D must independently show a broad floor below that same handle. The
    current floor cloud refines the lateral centre; no scene pose, task site,
    or simulator state is used.
    """
    points=np.asarray(points,dtype=np.float64)
    inward=-np.asarray(handle.outward_world,dtype=np.float64).copy();inward[2]=0.
    norm=np.linalg.norm(inward)
    if norm<.95:raise ValueError('drawer normal is not horizontal')
    inward/=norm
    up=np.array([0.,0.,1.]);axis=np.cross(inward,up)
    axes=np.column_stack((axis,inward,up))
    anchor=np.asarray(handle.point_world,dtype=np.float64)
    local=(points-anchor)@axes
    selected=(abs(local[:,0])<.15)&(local[:,1]>.03)&(local[:,1]<.20)&(local[:,2]>-.045)&(local[:,2]<-.010)
    floor_points=points[selected];floor_local=local[selected]
    if len(floor_points)<100:raise ValueError('drawer floor is not visible below its handle')
    bins,counts=np.unique(np.rint(floor_points[:,2]/.002).astype(int),return_counts=True)
    band=abs(floor_points[:,2]-bins[np.argmax(counts)]*.002)<.002
    floor_points=floor_points[band];floor_local=floor_local[band]
    low,high=np.quantile(floor_local[:,:2],[.01,.99],axis=0)
    if len(floor_points)<100 or not .14<high[0]-low[0]<.27 or high[1]-low[1]<.06:
        raise ValueError('drawer floor lacks a broad lateral/depth patch')
    lateral_center=float((low[0]+high[0])/2.)
    if abs(lateral_center)>.05:raise ValueError('floor does not align with the handle')
    lower=np.array([lateral_center-.205/2+.004,.03678+.004])
    upper=np.array([lateral_center+.205/2-.004,.18882-.004])
    vertical=int(np.argmax(abs(source.axes_world[2])))
    planar=[i for i in range(3) if i!=vertical]
    if ('bowl' in source.name.split() and abs(source.axes_world[2,vertical])>.95
            and max(source.extents_m[planar])/min(source.extents_m[planar])<1.20):
        payload_half=np.full(2,min(source.extents_m[planar])/2.)
    else:
        payload_half=abs(axes[:,:2].T@source.axes_world)@(source.extents_m/2.)
    safe_lower=lower+payload_half+.003;safe_upper=upper-payload_half-.003
    if np.any(safe_lower>safe_upper):raise ValueError('payload does not fit the drawer floor')
    requested=((preferred.centroid_world-anchor)@axes)[:2]
    centre_xy=np.clip(requested,safe_lower,safe_upper)
    half_xy=np.minimum(centre_xy-lower,upper-centre_xy)
    floor=float(np.median(floor_points[:,2]))
    if require_grounded_height and abs(floor-preferred.bounds_min_world[2])>.010:
        raise ValueError('handle support disagrees with the grounded drawer floor')
    height=.055
    center=anchor+axes[:,:2]@centre_xy;center[2]=floor+height/2.
    extents=np.r_[2.*half_xy,height];half_world=abs(axes)@(extents/2.)
    return SceneObject(preferred.name,center,axes,extents,center-half_world,center+half_world,
                       min(preferred.confidence,handle.confidence),len(floor_points),
                       surface_points_world=floor_points)


def observed_drawer_interior(observation, label, preferred, source, *, reference=None):
    from ..common import CameraCalibration,CameraFrame
    from ..goal_skills.detectors import DrawerHandleDetector
    from ..perception.adapters import coerce_rgbd_frame
    from ..perception.geometry import backproject_frame

    levels=[word for word in label.split() if word in {'top','middle','bottom'}]
    if len(levels)!=1:raise LookupError('drawer level is not explicit')
    frames=[coerce_rgbd_frame(frame,name=name) for name,frame in observation.cameras.items()]
    if len(frames)!=2:raise LookupError('drawer placement requires both public camera frames')
    cameras={f.name:CameraFrame(f.rgb,f.depth_m,CameraCalibration(f.name,f.rgb.shape[1],f.rgb.shape[0],
                f.intrinsics,f.world_from_camera,f.observation_v_flipped)) for f in frames}
    points=np.concatenate([backproject_frame(f,stride=2).points_world for f in frames])
    detector=DrawerHandleDetector();view=SimpleNamespace(cameras=cameras)
    if reference is not None:
        handle=detector.track(view,reference,levels[0])
        return drawer_interior_from_points(points,handle,preferred,source)
    proposals=[]
    for rank in ('top','middle','bottom'):
        try:
            handle=detector.detect(view,rank)
        except LookupError:
            continue
        above_floor=handle.point_world[2]-preferred.bounds_min_world[2]
        if not .012<above_floor<.050:
            continue
        try:
            region=drawer_interior_from_points(points,handle,preferred,source)
        except ValueError:
            continue
        proposals.append((abs(above_floor-.029),region))
    if not proposals:
        # A front/handle language crop can seed the tabletop height. If its
        # floor cannot bind any handle, use the explicitly requested level
        # and require a fresh broad floor directly below that visible bar.
        # Keep the measured wall/footprint gates; only the rejected crop's
        # height agreement is superseded by this independent surface.
        handle=detector.detect(view,levels[0])
        return drawer_interior_from_points(
            points,handle,preferred,source,require_grounded_height=False,
        )
    return min(proposals,key=lambda p:p[0])[1]


def observed_handle_drawer_interior(observation, label, source, *, preferred=None, reference=None):
    """Recover a visible floor when the language crop cannot provide one.

    The requested drawer level binds the public handle detector. Its existing
    physical wall model bounds the floor, while current depth must independently
    show a broad support below the same handle. An absent language crop uses
    the middle of those walls as the preferred location; it never supplies a
    floor height. Valid language-grounded floors keep their existing path.
    """
    from ..common import CameraCalibration, CameraFrame
    from ..goal_skills.detectors import DrawerHandleDetector
    from ..perception.adapters import coerce_rgbd_frame
    from ..perception.geometry import backproject_frame

    levels=[word for word in label.split() if word in {'top','middle','bottom'}]
    if len(levels)!=1:raise LookupError('drawer level is not explicit')
    if not {'agentview','wrist'}<=set(observation.cameras):
        raise LookupError('drawer floor recovery requires both public camera frames')
    frames=[coerce_rgbd_frame(observation.cameras[name],name=name) for name in ('agentview','wrist')]
    cameras={f.name:CameraFrame(f.rgb,f.depth_m,CameraCalibration(f.name,f.rgb.shape[1],f.rgb.shape[0],
                f.intrinsics,f.world_from_camera,f.observation_v_flipped)) for f in frames}
    view=SimpleNamespace(cameras=cameras);detector=DrawerHandleDetector()
    handle=(detector.track(view,reference,levels[0]) if reference is not None
            else detector.detect(view,levels[0]))
    points=np.concatenate([backproject_frame(f,stride=2).points_world for f in frames])
    if preferred is None:
        inward=-np.asarray(handle.outward_world,dtype=float).copy();inward[2]=0.
        if np.linalg.norm(inward)<.95:raise ValueError('drawer normal is not horizontal')
        inward/=np.linalg.norm(inward)
        axes=np.column_stack((np.cross(inward,[0.,0.,1.]),inward,[0.,0.,1.]))
        # Same public handle-relative wall dimensions as drawer_interior_from_points.
        center=np.asarray(handle.point_world)+inward*((.03678+.18882)/2)
        extents=np.array([.205,.18882-.03678,.055]);half=abs(axes)@(extents/2)
        preferred=SceneObject(label,center,axes,extents,center-half,center+half,handle.confidence,100)
    return drawer_interior_from_points(points,handle,preferred,source,require_grounded_height=False)
