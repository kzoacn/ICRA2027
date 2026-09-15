"""Training-free LIBERO texture gallery and collision-shape priors.

The gallery is built only from static public task assets.  It never reads the
active simulator model, BDDL, object poses, or instance segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from .schema import ObjectInstance, normalize_label


FloatArray = NDArray[np.float64]

DEFAULT_ASSET_ROOT = Path(os.environ.get(
    "LIBERO_ASSET_ROOT",
    str(Path(__file__).resolve().parents[3] / "resources" / "assets"),
)).expanduser()

LIBERO_OBJECT_LABELS = frozenset(
    {
        "alphabet soup",
        "cream cheese",
        "salad dressing",
        "bbq sauce",
        "ketchup",
        "tomato sauce",
        "butter",
        "milk",
        "chocolate pudding",
        "orange juice",
    }
)

# LIBERO-Object's ten sources plus its receptacle.  Aliases intentionally use
# natural-language task spelling, not simulator instance names.
OBJECT_GALLERY_SPECS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "alphabet soup": ("stable_hope_objects", "alphabet_soup", "texture_map.png"),
        "cream cheese": ("stable_hope_objects", "cream_cheese", "texture_map.png"),
        "salad dressing": ("stable_hope_objects", "salad_dressing", "texture_map.png"),
        "bbq sauce": ("stable_hope_objects", "bbq_sauce", "texture_map.png"),
        "ketchup": ("stable_hope_objects", "ketchup", "texture_map.png"),
        "tomato sauce": ("stable_hope_objects", "tomato_sauce", "texture_map.png"),
        "butter": ("stable_hope_objects", "butter", "texture_map.png"),
        "milk": ("stable_hope_objects", "milk", "texture_map.png"),
        "chocolate pudding": ("stable_hope_objects", "chocolate_pudding", "texture_map.png"),
        "orange juice": ("stable_hope_objects", "orange_juice", "texture_map.png"),
        "basket": ("stable_scanned_objects", "basket", "texture.png"),
        "black bowl": ("stable_scanned_objects", "akita_black_bowl", "texture.png"),
        "plate": ("stable_scanned_objects", "plate", "texture.png"),
        "ramekin": (
            "stable_scanned_objects",
            "glazed_rim_porcelain_ramekin",
            "texture.png",
        ),
        "cookie box": ("stable_hope_objects", "cookies", "texture_map.png"),
    }
)


@dataclass(frozen=True)
class PublicAssetGallerySpec:
    """One immutable, public LIBERO asset used as an appearance/size prior.

    ``texture_names`` are relative to the asset directory.  A few articulated
    assets use both a texture and a constant material colour; those colours are
    represented explicitly instead of rendering simulator state.  Every value
    below comes from the checked-in public asset XML/MTL, never the active
    MuJoCo model or an episode description.
    """

    collection: str
    folder: str
    xml_name: str
    texture_names: tuple[str, ...]
    material_rgb: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.collection or not self.xml_name:
            raise ValueError("asset collection and XML name are required")
        if not self.texture_names and not self.material_rgb:
            raise ValueError("an asset prior needs a texture or public material colour")
        for colour in self.material_rgb:
            if len(colour) != 3 or any(not 0 <= int(channel) <= 255 for channel in colour):
                raise ValueError("material RGB values must contain three uint8 channels")


def _legacy_asset_specs() -> dict[str, PublicAssetGallerySpec]:
    specs: dict[str, PublicAssetGallerySpec] = {}
    for label, (collection, folder, texture_name) in OBJECT_GALLERY_SPECS.items():
        xml_name = f"{folder}.xml"
        if label == "black bowl":
            xml_name = "akita_black_bowl.xml"
        elif label == "cookie box":
            xml_name = "cookies.xml"
        specs[label] = PublicAssetGallerySpec(
            collection,
            folder,
            xml_name,
            (texture_name,),
        )
    return specs


# Additional public assets used throughout Goal/90/10.  Prototype labels name
# the physical appearance, whereas instruction labels may be broader (for
# example either black or yellow books satisfy ``book``).  The query mapping
# below keeps those two concepts separate.
_PUBLIC_ASSET_GALLERY_SPECS = _legacy_asset_specs()
_PUBLIC_ASSET_GALLERY_SPECS.update(
    {
        "red bowl": PublicAssetGallerySpec(
            "stable_scanned_objects", "red_bowl", "red_bowl.xml", ("texture.png",)
        ),
        "white bowl": PublicAssetGallerySpec(
            "stable_scanned_objects", "white_bowl", "white_bowl.xml", ("texture.png",)
        ),
        "red mug": PublicAssetGallerySpec(
            "turbosquid_objects",
            "red_coffee_mug",
            "red_coffee_mug.xml",
            ("red_coffee_mug_texture.png",),
        ),
        "white mug": PublicAssetGallerySpec(
            "turbosquid_objects",
            "porcelain_mug",
            "porcelain_mug.xml",
            ("porcelain_mug_texture.png",),
        ),
        "yellow and white mug": PublicAssetGallerySpec(
            "turbosquid_objects",
            "white_yellow_mug",
            "white_yellow_mug.xml",
            ("white_yellow_mug_texture.png",),
        ),
        "moka pot": PublicAssetGallerySpec(
            "turbosquid_objects",
            "moka_pot",
            "moka_pot.xml",
            ("metal_diff.png", "rubber_black.png"),
        ),
        "black book": PublicAssetGallerySpec(
            "turbosquid_objects",
            "black_book",
            "black_book.xml",
            ("black_book_texture.png",),
        ),
        "yellow book": PublicAssetGallerySpec(
            "turbosquid_objects",
            "yellow_book",
            "yellow_book.xml",
            ("yellow_book.png",),
        ),
        "frying pan": PublicAssetGallerySpec(
            "stable_scanned_objects",
            "chefmate_8_frypan",
            "chefmate_8_frypan.xml",
            ("texture.png",),
        ),
        "tray": PublicAssetGallerySpec(
            "turbosquid_objects",
            "wooden_tray",
            "wooden_tray.xml",
            ("crate_mat_BaseColor.png",),
        ),
        "caddy": PublicAssetGallerySpec(
            "turbosquid_objects",
            "desk_caddy",
            "desk_caddy.xml",
            ("desk_caddy_texture.png",),
        ),
        "wine rack": PublicAssetGallerySpec(
            "turbosquid_objects",
            "wine_rack",
            "wine_rack.xml",
            ("mahogany.png", "light-wood.png"),
            ((0, 88, 139),),
        ),
        "rack": PublicAssetGallerySpec(
            "stable_scanned_objects",
            "simple_rack",
            "simple_rack.xml",
            (),
            ((204, 204, 204),),
        ),
        "wine bottle": PublicAssetGallerySpec(
            "turbosquid_objects",
            "wine_bottle",
            "wine_bottle.xml",
            ("label_wine.png", "cork_texture.png"),
            ((1, 9, 1),),
        ),
        "shelf": PublicAssetGallerySpec(
            "turbosquid_objects",
            "wooden_shelf",
            "wooden_shelf.xml",
            ("dark_fine_wood.png",),
        ),
        "cabinet shelf": PublicAssetGallerySpec(
            "turbosquid_objects",
            "wooden_two_layer_shelf",
            "wooden_two_layer_shelf.xml",
            ("dark_fine_wood.png",),
        ),
        "microwave": PublicAssetGallerySpec(
            "articulated_objects",
            "",
            "microwave.xml",
            ("textures/metal1.png",),
            ((51, 51, 51),),
        ),
        "cabinet": PublicAssetGallerySpec(
            "articulated_objects",
            "",
            "wooden_cabinet.xml",
            ("wooden_cabinet/dark_fine_wood.png",),
        ),
        "drawer": PublicAssetGallerySpec(
            "articulated_objects",
            "wooden_cabinet",
            "wooden_cabinet_top/wooden_cabinet_top.xml",
            ("dark_fine_wood.png",),
        ),
        "stove": PublicAssetGallerySpec(
            "articulated_objects",
            "",
            "flat_stove.xml",
            ("flat_stove/metal.png", "flat_stove/button_dark_texture.png"),
        ),
    }
)
PUBLIC_ASSET_GALLERY_SPECS: Mapping[str, PublicAssetGallerySpec] = MappingProxyType(
    _PUBLIC_ASSET_GALLERY_SPECS
)


@dataclass(frozen=True)
class GalleryQueryPrior:
    """Meaningful multi-class gallery evidence for one instruction label."""

    family: tuple[str, ...]
    accepted: tuple[str, ...]
    primary: str

    def __post_init__(self) -> None:
        family = tuple(dict.fromkeys(normalize_label(label) for label in self.family))
        accepted = tuple(dict.fromkeys(normalize_label(label) for label in self.accepted))
        primary = normalize_label(self.primary)
        if len(family) < 2:
            raise ValueError("gallery query families must contain meaningful alternatives")
        if not accepted or not set(accepted) <= set(family) or primary not in accepted:
            raise ValueError("accepted/primary gallery labels must belong to the family")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "primary", primary)


_BOWL_FAMILY = ("red bowl", "black bowl", "white bowl", "ramekin", "plate")
_MUG_FAMILY = ("red mug", "white mug", "yellow and white mug", "moka pot")
_BOOK_FAMILY = ("black book", "yellow book", "cookie box", "cream cheese")
_PAN_FAMILY = ("frying pan", "tray", "stove", "plate")
_CADDY_FAMILY = ("caddy", "tray", "cabinet shelf", "wine rack")
_RACK_FAMILY = ("rack", "wine rack", "shelf", "caddy")
_SHELF_FAMILY = ("shelf", "cabinet shelf", "cabinet", "caddy")
_APPLIANCE_FAMILY = ("microwave", "cabinet", "caddy", "stove")
_DRAWER_FAMILY = ("drawer", "cabinet", "cabinet shelf", "microwave")
_BOTTLE_FAMILY = ("wine bottle", "salad dressing", "orange juice", "moka pot")
_PACKAGE_FAMILY = ("cream cheese", "butter", "cookie box", "milk")


def _query_prior(
    family: tuple[str, ...],
    *accepted: str,
    primary: str | None = None,
) -> GalleryQueryPrior:
    values = tuple(accepted)
    return GalleryQueryPrior(family, values, primary or values[0])


# These are deliberately not one-class galleries.  A DINO crop must be
# compared with visually/geometrically plausible alternatives; the gallery is
# never allowed to manufacture a detection by itself for these labels.
GALLERY_QUERY_PRIORS: Mapping[str, GalleryQueryPrior] = MappingProxyType(
    {
        "black bowl": _query_prior(_BOWL_FAMILY, "black bowl"),
        "white bowl": _query_prior(_BOWL_FAMILY, "white bowl"),
        # LIBERO uses the generic instruction noun ``bowl`` for the public
        # Akita black-bowl asset as well as for colour-unspecified bowls.  Keep
        # actual bowl variants admissible while ramekin/plate remain meaningful
        # negative family members.  DINO must still ground a crop first.
        "bowl": _query_prior(
            _BOWL_FAMILY,
            "black bowl",
            "white bowl",
            "red bowl",
            primary="black bowl",
        ),
        "plate": _query_prior(_BOWL_FAMILY, "plate"),
        "ramekin": _query_prior(_BOWL_FAMILY, "ramekin"),
        "red mug": _query_prior(_MUG_FAMILY, "red mug"),
        "white mug": _query_prior(_MUG_FAMILY, "white mug"),
        "yellow and white mug": _query_prior(_MUG_FAMILY, "yellow and white mug"),
        "moka pot": _query_prior(_MUG_FAMILY, "moka pot"),
        "book": _query_prior(_BOOK_FAMILY, "black book", "yellow book"),
        "frying pan": _query_prior(_PAN_FAMILY, "frying pan"),
        "tray": _query_prior(_CADDY_FAMILY, "tray"),
        "caddy": _query_prior(_CADDY_FAMILY, "caddy"),
        "wine bottle": _query_prior(_BOTTLE_FAMILY, "wine bottle"),
        "wine rack": _query_prior(_RACK_FAMILY, "wine rack"),
        "rack": _query_prior(_RACK_FAMILY, "rack"),
        "shelf": _query_prior(_SHELF_FAMILY, "shelf"),
        "cabinet shelf": _query_prior(_SHELF_FAMILY, "cabinet shelf"),
        "microwave": _query_prior(_APPLIANCE_FAMILY, "microwave"),
        "cabinet": _query_prior(_APPLIANCE_FAMILY, "cabinet"),
        "wooden cabinet": _query_prior(
            _APPLIANCE_FAMILY, "cabinet", primary="cabinet"
        ),
        "drawer": _query_prior(_DRAWER_FAMILY, "drawer"),
        "stove": _query_prior(_APPLIANCE_FAMILY, "stove"),
        "cream cheese box": _query_prior(
            _PACKAGE_FAMILY, "cream cheese", primary="cream cheese"
        ),
    }
)


def rgb_to_hsv(rgb: NDArray[np.uint8]) -> FloatArray:
    """Vectorised OpenCV-compatible HSV in the unit cube."""

    values = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    red, green, blue = values.T
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    active = delta > 1e-12
    red_max = active & (maximum == red)
    green_max = active & (maximum == green)
    blue_max = active & (maximum == blue)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-12)
    return np.column_stack((hue, saturation, maximum))


def hsv_histogram(
    rgb: NDArray[np.uint8],
    *,
    bins: tuple[int, int, int] = (12, 5, 5),
    weights: FloatArray | None = None,
) -> FloatArray:
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("rgb samples must have shape (N, 3) and dtype uint8")
    if len(pixels) == 0:
        raise ValueError("cannot histogram an empty RGB sample")
    histogram, _ = np.histogramdd(
        rgb_to_hsv(pixels),
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        weights=weights,
    )
    flat = histogram.reshape(-1).astype(np.float64)
    flat += 1e-9
    return flat / flat.sum()


def _quaternion_matrix(raw: str | None) -> FloatArray:
    if raw is None:
        return np.eye(3, dtype=np.float64)
    quaternion = np.fromstring(raw, sep=" ", dtype=np.float64)
    if quaternion.shape != (4,) or float(np.linalg.norm(quaternion)) < 1e-10:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _euler_matrix(raw: str | None) -> FloatArray:
    """Return MuJoCo's default intrinsic xyz Euler rotation."""

    if raw is None:
        return np.eye(3, dtype=np.float64)
    angles = np.fromstring(raw, sep=" ", dtype=np.float64)
    if angles.shape != (3,):
        return np.eye(3, dtype=np.float64)
    x, y, z = angles
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rx @ ry @ rz


def _element_rotation(element: ET.Element) -> FloatArray:
    if element.get("quat") is not None:
        return _quaternion_matrix(element.get("quat"))
    return _euler_matrix(element.get("euler"))


def collision_aabb_dimensions(xml_path: str | Path) -> FloatArray:
    """Transform group-0 primitive corners and return their aggregate local AABB.

    This deliberately ignores often-inaccurate top/bottom sites.  In
    particular, the chocolate-pudding collision box is rotated onto its side;
    transforming its corners recovers its roughly 2.7 cm vertical height.
    """

    root = ET.parse(Path(xml_path)).getroot()
    all_corners: list[FloatArray] = []
    signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    def position(element: ET.Element) -> FloatArray:
        value = np.fromstring(element.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
        return value if value.shape == (3,) else np.zeros(3, dtype=np.float64)

    def visit(
        element: ET.Element,
        parent_rotation: FloatArray,
        parent_translation: FloatArray,
    ) -> None:
        local_rotation = _element_rotation(element) if element.tag == "body" else np.eye(3)
        local_translation = position(element) if element.tag == "body" else np.zeros(3)
        body_rotation = parent_rotation @ local_rotation
        body_translation = parent_translation + parent_rotation @ local_translation
        for geom in element.findall("geom"):
            if geom.get("group", "0") != "0":
                continue
            kind = geom.get("type", "sphere")
            if kind == "mesh":
                continue
            size = np.fromstring(geom.get("size", ""), sep=" ", dtype=np.float64)
            if kind == "box" and size.shape == (3,):
                half = size
            elif kind in {"cylinder", "capsule"} and size.shape == (2,):
                half = np.array([size[0], size[0], size[1]], dtype=np.float64)
            elif kind == "sphere" and size.shape == (1,):
                half = np.repeat(size[0], 3)
            elif kind == "ellipsoid" and size.shape == (3,):
                half = size
            else:
                continue
            geom_rotation = body_rotation @ _element_rotation(geom)
            geom_translation = body_translation + body_rotation @ position(geom)
            all_corners.append(signs * half @ geom_rotation.T + geom_translation)
        for body in element.findall("body"):
            visit(body, body_rotation, body_translation)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        visit(worldbody, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    if not all_corners:
        raise ValueError(f"no group-0 collision primitives found in {xml_path}")
    corners = np.concatenate(all_corners, axis=0)
    return corners.max(axis=0) - corners.min(axis=0)


def _texture_pixels(path: Path, *, max_samples: int = 250_000) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a transitive dependency
        raise RuntimeError("Pillow is required to load the static texture gallery") from exc
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    pixels = rgba[..., :3].reshape(-1, 3)
    alpha = rgba[..., 3].reshape(-1)
    pixels = pixels[alpha > 16]
    if len(pixels) > max_samples:
        indices = np.linspace(0, len(pixels) - 1, max_samples, dtype=np.int64)
        pixels = pixels[indices]
    return pixels.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class AssetPrototype:
    label: str
    histogram: FloatArray
    dimensions_xyz_m: FloatArray
    texture_path: Path
    xml_path: Path

    def __post_init__(self) -> None:
        histogram = np.asarray(self.histogram, dtype=np.float64).reshape(-1)
        dimensions = np.asarray(self.dimensions_xyz_m, dtype=np.float64)
        if dimensions.shape != (3,) or np.any(dimensions <= 0):
            raise ValueError("asset dimensions must be a positive 3-vector")
        if np.any(histogram < 0) or histogram.sum() <= 0:
            raise ValueError("asset histogram is invalid")
        object.__setattr__(self, "label", normalize_label(self.label))
        object.__setattr__(self, "histogram", histogram / histogram.sum())
        object.__setattr__(self, "dimensions_xyz_m", dimensions.copy())


@dataclass(frozen=True)
class Classification:
    scores: Mapping[str, float]
    label: str
    confidence: float
    dimensions_xyz_m: FloatArray


class TextureSizeGallery:
    """HSV appearance plus XML collision dimensions; no learned parameters."""

    def __init__(
        self,
        prototypes: Mapping[str, AssetPrototype],
        *,
        color_weight: float = 0.40,
        shape_weight: float = 0.60,
        softmax_temperature: float = 0.13,
    ) -> None:
        if not prototypes:
            raise ValueError("at least one asset prototype is required")
        if color_weight < 0 or shape_weight < 0 or color_weight + shape_weight <= 0:
            raise ValueError("classification weights must be non-negative and non-zero")
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        self.prototypes = MappingProxyType(
            {normalize_label(label): prototype for label, prototype in prototypes.items()}
        )
        self.color_weight = float(color_weight / (color_weight + shape_weight))
        self.shape_weight = float(shape_weight / (color_weight + shape_weight))
        self.softmax_temperature = float(softmax_temperature)

    @classmethod
    def from_libero_assets(
        cls,
        asset_root: str | Path = DEFAULT_ASSET_ROOT,
        *,
        labels: Iterable[str] | None = None,
        **kwargs,
    ) -> "TextureSizeGallery":
        root = Path(asset_root)
        requested = None if labels is None else {normalize_label(label) for label in labels}
        prototypes: dict[str, AssetPrototype] = {}
        for label, spec in PUBLIC_ASSET_GALLERY_SPECS.items():
            if requested is not None and label not in requested:
                continue
            directory = root / spec.collection / spec.folder
            texture_paths = tuple(directory / name for name in spec.texture_names)
            xml_path = directory / spec.xml_name
            if not xml_path.is_file() or any(not path.is_file() for path in texture_paths):
                raise FileNotFoundError(f"incomplete LIBERO asset prototype: {directory}")
            histograms = [hsv_histogram(_texture_pixels(path)) for path in texture_paths]
            histograms.extend(
                hsv_histogram(
                    np.repeat(np.asarray(colour, dtype=np.uint8)[None, :], 256, axis=0)
                )
                for colour in spec.material_rgb
            )
            histogram = np.mean(np.stack(histograms), axis=0)
            prototypes[label] = AssetPrototype(
                label=label,
                histogram=histogram,
                dimensions_xyz_m=collision_aabb_dimensions(xml_path),
                # Compatibility field: material-only assets point at their
                # public XML, which is the source of the constant RGB prior.
                texture_path=texture_paths[0] if texture_paths else xml_path,
                xml_path=xml_path,
            )
        if requested is not None:
            missing = requested - set(prototypes)
            if missing:
                raise KeyError(f"unsupported gallery labels: {sorted(missing)}")
        return cls(prototypes, **kwargs)

    @staticmethod
    def orient_dimensions(
        prototype_dims: FloatArray,
        observed_extents: FloatArray,
    ) -> FloatArray:
        """Choose the static collision-box permutation matching observed gravity.

        LIBERO rotates several HOPE packages by 90 degrees when placing them
        upright, while butter / cream-cheese / chocolate-pudding remain in the
        XML orientation.  Trying the six rigid axis permutations uses only the
        static dimensions and measured OBB; it does not consult task state.
        """

        observed = np.asarray(observed_extents, dtype=np.float64)
        target = np.array([*sorted(observed[:2], reverse=True), observed[2]])
        candidates = [np.asarray(order, dtype=np.float64) for order in permutations(prototype_dims)]
        return min(
            candidates,
            key=lambda order: float(
                np.mean(np.abs(np.log(np.maximum(target, 0.004) / np.maximum(order, 0.004))))
            ),
        ).copy()

    @classmethod
    def _shape_similarity(cls, observed_extents: FloatArray, prototype_dims: FloatArray) -> float:
        observed = np.asarray(observed_extents, dtype=np.float64)
        observed_ordered = np.array([*sorted(observed[:2], reverse=True), observed[2]])
        prototype_ordered = cls.orient_dimensions(prototype_dims, observed)
        # Visible-surface extents can underestimate a side.  Log ratios keep the
        # metric scale invariant while clipping severe occlusion penalties.
        ratio_error = np.minimum(np.abs(np.log(np.maximum(observed_ordered, 0.004) / prototype_ordered)), 1.8)
        return float(np.exp(-np.mean(ratio_error)))

    def classify_features(
        self,
        histogram: FloatArray,
        observed_extents_m: FloatArray,
        *,
        allowed_labels: Sequence[str] | None = None,
    ) -> Classification:
        histogram = np.asarray(histogram, dtype=np.float64).reshape(-1)
        if histogram.sum() <= 0:
            raise ValueError("observed histogram cannot be empty")
        histogram = histogram / histogram.sum()
        labels = list(self.prototypes)
        if allowed_labels is not None:
            requested = {normalize_label(label) for label in allowed_labels}
            labels = [label for label in labels if label in requested]
            missing = requested - set(labels)
            if missing:
                raise KeyError(f"unsupported gallery labels: {sorted(missing)}")
        if not labels:
            raise ValueError("allowed_labels selected no prototypes")
        logits: list[float] = []
        footprint = float(np.max(np.asarray(observed_extents_m)[:2]))
        for label in labels:
            prototype = self.prototypes[label]
            if prototype.histogram.shape != histogram.shape:
                raise ValueError("observed and gallery histogram bins differ")
            color = float(np.sum(np.sqrt(histogram * prototype.histogram)))
            shape = self._shape_similarity(observed_extents_m, prototype.dimensions_xyz_m)
            score = self.color_weight * color + self.shape_weight * shape
            # Basket is an order of magnitude larger in footprint than the ten
            # graspable packages.  This observable geometric gate prevents a
            # brown/white package from inheriting the receptacle label.
            if label == "basket":
                score += 0.22 if footprint >= 0.105 else -0.25
            elif footprint >= 0.125:
                score -= 0.15
            logits.append(score / self.softmax_temperature)
        logits_array = np.asarray(logits, dtype=np.float64)
        probabilities = np.exp(logits_array - logits_array.max())
        probabilities /= probabilities.sum()
        scores = {label: float(score) for label, score in zip(labels, probabilities, strict=True)}
        best = max(scores, key=scores.__getitem__)
        return Classification(
            scores=MappingProxyType(scores),
            label=best,
            confidence=float(scores[best]),
            dimensions_xyz_m=self.orient_dimensions(
                self.prototypes[best].dimensions_xyz_m,
                observed_extents_m,
            ),
        )

    def classify(
        self,
        instance: ObjectInstance,
        *,
        allowed_labels: Sequence[str] | None = None,
    ) -> Classification:
        if instance.color_histogram is None:
            raise ValueError("instance has no RGB histogram")
        return self.classify_features(
            instance.color_histogram,
            instance.observed.extents_m,
            allowed_labels=allowed_labels,
        )
