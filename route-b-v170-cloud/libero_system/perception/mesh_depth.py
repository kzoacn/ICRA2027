"""Perspective depth rasterization for public rigid mesh hypotheses."""
import numpy as np

def render_depth(triangles, frame):
    local=(triangles-frame.world_from_camera[:3,3])@frame.world_from_camera[:3,:3]
    intr=frame.intrinsics
    uv=local[...,:2]/local[...,2:]*[intr[0,0],intr[1,1]]+[intr[0,2],intr[1,2]]
    if frame.observation_v_flipped:uv[...,1]=frame.height-1-uv[...,1]
    depth=np.full((frame.height,frame.width),np.inf)
    for p,z in zip(uv,local[...,2]):
        if min(z)<=.05:continue
        lower=np.maximum(np.ceil(p.min(0)).astype(int),0)
        upper=np.minimum(np.floor(p.max(0)).astype(int),[frame.width-1,frame.height-1])
        if np.any(upper<lower):continue
        a,b,c=p;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-8:continue
        x,y=np.meshgrid(np.arange(lower[0],upper[0]+1),np.arange(lower[1],upper[1]+1))
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        w=1-u-v;mask=(u>=0)&(v>=0)&(w>=0)
        interpolated=1/(u/z[0]+v/z[1]+w/z[2])
        patch=depth[lower[1]:upper[1]+1,lower[0]:upper[0]+1]
        patch[mask]=np.minimum(patch[mask],interpolated[mask])
    return depth

