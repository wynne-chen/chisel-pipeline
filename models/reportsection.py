from dataclasses import dataclass, field
from matplotlib.figure import Figure
from typing import Any

#--- REPORT SECTION ---#

@dataclass
class ReportSection:
    idx: int
    slug: str
    title: str
    text_lines: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[Figure] = field(default_factory=list) 
    include_in_pdf: bool = True
    include_images: bool = True
    include_text: bool = True
    show_inline_override: bool | None = None