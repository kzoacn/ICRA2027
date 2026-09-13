"""Frozen perception component assembly for both control routes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from libero_system.perception import (
    BoxGeometryExtractor,
    DEFAULT_ASSET_ROOT,
    DEFAULT_GROUNDING_DINO_TINY,
    GroundingDINOBoxDetector,
    ObjectInstance,
    PerceptionConfig,
    RouteBScenePerceptionAdapter,
    RouteCSceneEstimatorAdapter,
    SensorOnlyScenePerception,
    TextureSizeGallery,
)

if TYPE_CHECKING:
    from libero_system.integration.config import EvaluationConfig


class LabelSpecificRouteCSceneEstimatorAdapter(RouteCSceneEstimatorAdapter):
    """Keep requested-label evidence separate from a component's top class.

    The neutral gallery deliberately retains close alternatives so downstream
    language grounding can compare them.  A tracked ``ketchup`` hypothesis
    must therefore inherit the component's ketchup score, not the confidence
    of an unrelated top class such as milk.  Route C resolves same-label
    candidates by ``SceneEntity.confidence``, so both fields must remain tied
    to the requested label at this policy boundary.
    """

    def __init__(self, *args, enable_episode_cache: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enable_episode_cache = bool(enable_episode_cache)
        self._scene_cache: dict[tuple[str, ...], Any] = {}

    def reset(self) -> None:
        super().reset()
        self._scene_cache.clear()

    def invalidate_sensor_cache(self) -> None:
        """Force the next estimate to use newly captured RGB-D frames."""

        self._scene_cache.clear()

    def observe(self, requested_labels):
        return self._observe_label_specific(
            requested_labels,
            source_reference=None,
        )

    def observe_source_selector(self, requested_labels, source_reference):
        """Cache one explicitly scoped, language-derived source observation."""

        return self._observe_label_specific(
            requested_labels,
            source_reference=source_reference,
        )

    def _observe_label_specific(self, requested_labels, *, source_reference):
        normalized_requested = {
            " ".join(str(label).lower().replace("_", " ").split())
            for label in requested_labels
        }
        selector_key: tuple[str, ...] = ()
        if source_reference is not None:
            from libero_system.route_c.schema import EntityRef

            if not isinstance(source_reference, EntityRef):
                raise TypeError("source selector observation requires an EntityRef")
            selector = source_reference.selector
            selector_key = (
                "__source_selector__",
                source_reference.role,
                " ".join(source_reference.label.lower().replace("_", " ").split()),
                selector.relation.value if selector is not None else "",
                *(selector.references if selector is not None else ()),
            )
        key = (*tuple(sorted(normalized_requested)), *selector_key)
        if self.enable_episode_cache and key in self._scene_cache:
            return self._scene_cache[key]
        scene = (
            super().observe_source_selector(requested_labels, source_reference)
            if source_reference is not None
            else super().observe(requested_labels)
        )
        # Generic neutral components remain represented in ``obstacle_sdf``,
        # but a requested semantic label must be resolved only through the
        # gated, label-specific tracks.  Otherwise a high-confidence tabletop
        # generic bowl can steal an ON/CENTER/NEXT_TO instruction from the
        # DINO+gallery track of the actual raised bowl.
        scene = replace(
            scene,
            entities=tuple(
                entity
                for entity in scene.entities
                if entity.label not in normalized_requested
                or entity.instance_id.startswith("tracked-")
            ),
        )
        if self.enable_episode_cache:
            self._scene_cache[key] = scene
        return scene

    @staticmethod
    def _clone_track(track_id: str, label: str, item: ObjectInstance) -> ObjectInstance:
        label_score = float(item.score_for(label))
        center = item.center_world.copy()
        grasp = item.grasp_point_world.copy()
        if " ".join(label.lower().replace("_", " ").split()) == "plate":
            center[:2] = item.observed.center_world[:2]
            grasp[:2] = item.observed.center_world[:2]
        return ObjectInstance(
            instance_id=track_id,
            center_world=center,
            grasp_point_world=grasp,
            axes_world=item.axes_world,
            extents_m=item.extents_m,
            observed=item.observed,
            label_scores={label: label_score},
            confidence=label_score,
            point_count=item.point_count,
            color_histogram=item.color_histogram,
        )


@dataclass(frozen=True)
class PerceptionFactoryContext:
    """Sanitised deployment settings exposed to a custom perception factory.

    Benchmark identity, task schedule, seed, output paths, and evaluator
    settings are intentionally not representable here.  A factory can select
    a frozen model/device and expected image size, but cannot branch on the
    suite or task being scored.
    """

    device: str
    perception_backend: str
    perception_model: Path | None
    image_size: int

    @classmethod
    def from_evaluation_config(
        cls, config: EvaluationConfig
    ) -> "PerceptionFactoryContext":
        return cls(
            device=config.device,
            perception_backend=config.perception_backend,
            perception_model=config.perception_model,
            image_size=config.image_size,
        )


def load_callable(specification: str) -> Callable[..., Any]:
    """Load an explicit ``module:callable`` without evaluating arbitrary text."""

    if ":" not in specification:
        raise ValueError("factory override must use module:callable syntax")
    module_name, attribute_name = specification.split(":", maxsplit=1)
    if not module_name or not attribute_name:
        raise ValueError("factory override must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"{specification!r} does not resolve to a callable")
    return factory


@dataclass
class PerceptionBundle:
    """Share one frozen model across all 50 episodes in a process."""

    backend: SensorOnlyScenePerception
    box_detector: Any | None = None
    box_extractor: BoxGeometryExtractor | None = None
    route_c_episode_cache: bool = False

    def for_route_b(self) -> RouteBScenePerceptionAdapter:
        return RouteBScenePerceptionAdapter(
            self.backend,
            box_detector=self.box_detector,
            box_extractor=self.box_extractor,
        )

    def for_route_c(self, frame_provider) -> RouteCSceneEstimatorAdapter:
        return LabelSpecificRouteCSceneEstimatorAdapter(
            frame_provider,
            self.backend,
            box_detector=self.box_detector,
            box_extractor=self.box_extractor,
            enable_episode_cache=self.route_c_episode_cache,
        )

    def close(self) -> None:
        # Hugging Face modules do not require close(), but custom detector
        # services may own a socket or worker process.
        close = getattr(self.box_detector, "close", None)
        if callable(close):
            close()


def unified_perception_config() -> PerceptionConfig:
    """One deployment workspace for every suite, without benchmark metadata.

    The broad z interval contains both LIBERO's low Object table and elevated
    kitchen worktop.  The support height itself is estimated from RGB-D in each
    episode.  A common support-gap gate keeps the Object gallery behaviour
    conservative; elevated Spatial sources are obtained from frozen DINO
    boxes and measured fixture-relative recovery rather than a suite hint.
    """

    return PerceptionConfig(max_support_gap_m=0.025)


def build_perception_bundle_from_context(
    context: PerceptionFactoryContext,
    *,
    perception_factory: str | None = None,
) -> PerceptionBundle:
    """Build perception from policy-safe deployment settings only.

    Formal Route C uses this entry point inside its isolated policy process.
    Keeping suite, task, init-state, seed, output paths, and evaluator objects
    out of the function signature makes accidental benchmark-identity
    delivery structurally impossible.
    """

    if type(context) is not PerceptionFactoryContext:
        raise TypeError("context must be an exact PerceptionFactoryContext")
    if perception_factory:
        value = load_callable(perception_factory)(context)
        if not isinstance(value, PerceptionBundle):
            raise TypeError("custom perception factory must return PerceptionBundle")
        return value

    asset_root = Path(DEFAULT_ASSET_ROOT)
    gallery = TextureSizeGallery.from_libero_assets(asset_root)
    backend = SensorOnlyScenePerception(
        config=unified_perception_config(),
        classifier=gallery,
    )
    # Backend choice is an explicit deployment setting, never inferred from a
    # benchmark suite label.  ``auto`` means the pinned local frozen detector.
    needs_detector = context.perception_backend in {"auto", "grounding-dino"}
    route_c_episode_cache = context.perception_backend != "gallery"
    if not needs_detector:
        return PerceptionBundle(
            backend,
            route_c_episode_cache=route_c_episode_cache,
        )

    model_path = context.perception_model or Path(DEFAULT_GROUNDING_DINO_TINY)
    if not model_path.exists():
        raise FileNotFoundError(
            "frozen Grounding-DINO snapshot is missing; pass --perception-model "
            "or use --perception-backend gallery for LIBERO-Object"
        )
    detector = GroundingDINOBoxDetector(
        model_name_or_path=model_path,
        device=context.device,
        local_files_only=True,
        workspace_bounds=(backend.config.workspace_min, backend.config.workspace_max),
    )
    return PerceptionBundle(
        backend,
        box_detector=detector,
        box_extractor=BoxGeometryExtractor(),
        route_c_episode_cache=route_c_episode_cache,
    )


def build_perception_bundle(config: EvaluationConfig) -> PerceptionBundle:
    context = PerceptionFactoryContext.from_evaluation_config(config)
    return build_perception_bundle_from_context(
        context,
        perception_factory=config.perception_factory,
    )
