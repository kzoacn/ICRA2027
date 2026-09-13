"""Register partial RGB-D shelf surfaces to a fixed public collision-box prior.

Only public box geoms are parsed; sites and evaluation regions are ignored.
The roof and one measured end wall anchor the template in the current frame.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from ..perception.gallery import DEFAULT_ASSET_ROOT, PUBLIC_ASSET_GALLERY_SPECS
from .models import SceneObject

_SPEC = PUBLIC_ASSET_GALLERY_SPECS["cabinet shelf"]
ASSET = DEFAULT_ASSET_ROOT / _SPEC.collection / _SPEC.folder / _SPEC.xml_name

def _fit(points, approach_from, table_height):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 500 or not np.all(np.isfinite(points)):
        raise ValueError("partial shelf requires a finite measured point cloud")
    boxes=[]
    for geom in ET.parse(ASSET).getroot().findall('.//geom'):
        if geom.get('type') != 'box' or geom.get('group') != '0': continue
        center=np.fromstring(geom.get('pos'),sep=' ')
        q=np.fromstring(geom.get('quat','1 0 0 0'),sep=' ')
        half=abs(Rotation.from_quat(q[[1,2,3,0]]).as_matrix())@np.fromstring(geom.get('size'),sep=' ')
        boxes.append((center-half,center+half))
    boards=sorted([b for b in boxes if b[1][2]-b[0][2]<.012 and min((b[1]-b[0])[:2])>.1],key=lambda b:b[1][2])
    middle,roof=boards[-2:]
    ends=sorted([b for b in boxes if b[1][2]-b[0][2]>.15 and b[1][0]-b[0][0]<.03],key=lambda b:b[0][0])
    edges=np.arange(points[:,2].min(),points[:,2].max()+.003,.003)
    counts,edges=np.histogram(points[:,2],edges)
    i=int(np.argmax(counts));height=(edges[i]+edges[i+1])/2
    top=points[abs(points[:,2]-height)<.003]
    low=points[points[:,2]<height-.035]
    if len(top)<300 or len(low)<150: raise ValueError('insufficient roof/side wall samples')
    wall_center=np.median(low,axis=0); _,_,vectors=np.linalg.svd(low-wall_center,full_matrices=False)
    normal=vectors[-1]
    if abs(normal[2])>.15:raise ValueError('side wall not vertical')
    normal[2]=0;normal/=np.linalg.norm(normal)
    error=np.quantile(abs((low-wall_center)@normal),.9)
    if error>.004:raise ValueError('lower crop is not a single side wall')
    if normal@(wall_center-np.median(top,axis=0))<0: normal*=-1
    outward=np.cross([0,0,1],normal)
    if outward@(approach_from-np.median(top,axis=0))<0:outward*=-1
    width=np.cross(outward,[0,0,1]);axes=np.column_stack((width,outward,[0,0,1]))
    local=top@axes;lo,hi=np.quantile(local,(.01,.99),axis=0)
    side=1 if normal@width>0 else 0
    side_face=ends[side][side][0]
    origin=np.zeros(3);origin[0]=np.median(low@width)-side_face
    origin[1]=(lo[1]+hi[1])/2-(roof[0][1]+roof[1][1])/2
    origin[2]=np.median(top[:,2])-roof[1][2]
    result={}
    for relation in ('upper_shelf','lower_shelf'):
        lower=np.array([ends[0][1][0]+.01,roof[0][1]+.015,middle[1][2] if relation=='upper_shelf' else table_height-origin[2]])
        upper=np.array([ends[-1][0][0]-.01,roof[1][1]-.015,roof[0][2]-.005 if relation=='upper_shelf' else middle[0][2]-.005])
        result[relation]=dict(center=(axes@(origin+(lower+upper)/2)).tolist(),axes=axes.tolist(),extents=(upper-lower).tolist())
    return dict(regions=result,side_residual90=error,origin=(axes@origin).tolist(),asset=str(ASSET))

def partial_shelf_region_from_points(points, relation, approach_from, support_height):
    if relation not in {"upper_shelf", "lower_shelf"}:
        raise ValueError("unknown shelf level")
    result = _fit(points, approach_from, support_height)["regions"][relation]
    center, axes, extent = (np.asarray(result[key]) for key in ("center", "axes", "extents"))
    if not np.all(np.isfinite(extent)) or min(extent) < .03 or max(extent) > .45:
        raise ValueError("partial shelf fit has inconsistent clearances")
    half = abs(axes) @ (extent / 2)
    return SceneObject("cabinet shelf", center, axes, extent, center - half, center + half,
                       .7, len(points), surface_points_world=points)
