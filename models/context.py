from dataclasses import dataclass, field
from typing import Any


@dataclass
class EDAContext:
    """Transient cache for report-time intermediate artefacts."""

    avg_img: Any = None
    contrast_heatmap: Any = None
    spatial_metadata: Any = None
    spatial_layer: Any = None
    zoom_level: Any = None
    bounds_area: Any = None
    obj_stats_all: Any = None
    foreground_classes: list[Any] | None = None
    _eligible_warmed: bool = False
    eligible_sample_indices_by_class: dict[Any, list[int]] = field(default_factory=dict)
    obj_stats_by_class: dict[Any, dict[str, Any]] = field(default_factory=dict)
    multiclass_summary: Any = None

    def as_dict(self) -> dict[str, Any]:
        """Backwards-compatible dict view for existing section code paths."""
        return {
            "avg_img": self.avg_img,
            "contrast_heatmap": self.contrast_heatmap,
            "spatial_metadata": self.spatial_metadata,
            "spatial_layer": self.spatial_layer,
            "zoom_level": self.zoom_level,
            "bounds_area": self.bounds_area,
            "obj_stats_all": self.obj_stats_all,
            "foreground_classes": self.foreground_classes,
            "_eligible_warmed": self._eligible_warmed,
            "eligible_sample_indices_by_class": self.eligible_sample_indices_by_class,
            "obj_stats_by_class": self.obj_stats_by_class,
            "multiclass_summary": self.multiclass_summary,
        }
