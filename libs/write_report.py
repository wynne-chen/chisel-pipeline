#--- IMPORTS ---#

import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from datetime import datetime
from models.reportsection import ReportSection


class ReportArtefactManager:
    def __init__(self, base_dir='reports', stage='pre', set_name: str = None):
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{stage}"
        self.stage = stage
        self.set_name = set_name
        self.run_dir = Path(base_dir) / set_name / run_id
        self.images_dir = self.run_dir / 'images'
        self.pdf_path = self.run_dir / f'report_{set_name}_{stage}-processing.pdf'
        self.summary_path = self.run_dir / 'summary.json'
    
    def ensure_output_dirs(self, *, save_pdf: bool, save_images: bool, save_summary: bool) -> None:
        '''
        Create on-disk layout only when at least one artefact will be written.
        '''
        if not (save_pdf or save_images or save_summary):
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

    def saved_image_path(self, idx: int, slug: str, variant: str | None = None) -> Path:
        '''
        Generates .png path for saved plot. Can specify variant (e.g. 'original', 'processed') to add to filename.
        '''

        base_name = f'{idx:02d}_{slug}'
        if variant:
            base_name += f'_{variant}'
        return self.images_dir / f'{base_name}.png'

    
    def save_summary(self, summary: dict):
        '''
        Saves summary dictionary to JSON file.
        '''
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
    

#--- WRITE REPORT ---#

'''
WriteReport is a class that writes a report to a pdf file.
'''

class WriteReport:

    A4_PORTRAIT_SIZE = (8.27, 11.69) # pdfpages uses inches
    A4_LANDSCAPE_SIZE = (11.69, 8.27)

    def __init__(self, artefacts: ReportArtefactManager, set_name: str):
        self.artefacts = artefacts
        self.stage = artefacts.stage
        self.set_name = set_name

    def _section_text(self, section: ReportSection) -> str:
        '''
        Returns the text for a report section.
        '''
        lines = []
        if section.text_lines:
            lines.extend(section.text_lines)
        if section.metrics:
            lines.append("")
            lines.append("Metrics:")
            for k, v in section.metrics.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines) if lines else "(no text)"
    
    def _save_fixed_pdf_page(self, pdf: PdfPages, fig: Figure):
        '''
        Saves a figure to a fixed A4 page in the PDF. Currently using landscape orientation.
        '''
        fig.set_size_inches(*self.A4_LANDSCAPE_SIZE, forward=True)
        fig.set_constrained_layout(False)
        pdf.savefig(fig)

    def _emit_section_header_inline(self, section: ReportSection):
        """Single source of section headers for inline mode."""
        print(f"\n{'='*40}\n{section.idx:02d}. {section.title}\n{'='*40}")
        sys.stdout.flush()

    def _build_text_figure(self, section: ReportSection) -> Figure:
        """Create figure page for section text payload."""
        text_fig = plt.figure(figsize=self.A4_LANDSCAPE_SIZE)
        text_fig.suptitle(f"{section.idx:02d}. {section.title}", fontsize=14, y=0.98)
        text_fig.text(
            0.06,
            0.94,
            self._section_text(section),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            wrap=True
        )
        plt.axis("off")
        return text_fig

    def _emit_figure(self, section: ReportSection, fig: Figure, variant: str, show_inline: bool, save_image: bool, save_pdf: bool, pdf: PdfPages | None):
        """Fan out one figure to enabled sinks in deterministic order."""
        if save_image:
            fig.savefig(
                self.artefacts.saved_image_path(section.idx, section.slug, variant),
                dpi=160,
                bbox_inches="tight"
            )
        if save_pdf and pdf is not None:
            self._save_fixed_pdf_page(pdf, fig)
        if show_inline:
            print(f"Showing figure {variant} for section {section.slug}")
            sys.stdout.flush()
            plt.figure(fig.number)
            plt.show()


    def render_section(self, section: ReportSection, pdf: PdfPages | None = None,
    show_inline: bool = True, save_pdf: bool = True, save_image: bool = True):
        '''
        Renders a report section.
        '''
        if not show_inline and not save_image and not (save_pdf and pdf is not None):
            return

        if show_inline:
            self._emit_section_header_inline(section)
        
        # text before images
        has_text_payload = bool(section.text_lines) or bool(section.metrics)
        
        if section.include_text and has_text_payload:
            if show_inline:
                payload = self._section_text(section)
                if payload:
                    print(payload)
                    sys.stdout.flush()
            needs_text_fig = save_image or (save_pdf and pdf is not None)
            if needs_text_fig:
                text_fig = self._build_text_figure(section)
                self._emit_figure(section, text_fig, "summary", show_inline, save_image, save_pdf, pdf)
            if not show_inline:
                plt.close(text_fig)
    
        for fig_idx, fig in enumerate(section.figures):
            if fig is None:
                continue

            variant = f"fig{fig_idx+1}"
            self._emit_figure(section, fig, variant, show_inline, save_image, save_pdf, pdf)
            if not show_inline:
                plt.close(fig)
        sys.stdout.flush()

    
    def render_report(self, sections: list[ReportSection], show_inline: bool = False, save_pdf: bool = True, save_images: bool = True, save_summary: bool = True):
        '''
        NOTE: The PDF will not be saved and cannot be overridden if save_pdf is False at the top level.

        Args:
            sections (list[ReportSection]): The sections to render.
            show_inline (bool): Whether to show the plots inline.
            save_pdf (bool): Whether to save the PDF.
            save_images (bool): Whether to save the images.
            save_summary (bool): Whether to save the summary JSON.
        '''
        print(f"Running EDA on raw dataset ({'pre-processing' if self.stage=='pre' else 'post-processing'}...)")
        sys.stdout.flush()

        self.artefacts.ensure_output_dirs(save_pdf=save_pdf, save_images=save_images, save_summary=save_summary)

        title = f"EDA Report for Dataset: {self.set_name.capitalize()} ({'PRE' if self.stage=='pre' else 'POST'}-PROCESSING)"
        
        ordered_sections = sorted(sections, key=lambda s: (s.idx, s.slug))

        if save_pdf:
            with PdfPages(self.artefacts.pdf_path) as pdf:
                # Title page
                title_fig = plt.figure(figsize=self.A4_LANDSCAPE_SIZE)
                title_fig.text(0.5, 0.7, title, ha='center', va='center', fontsize=20, weight='bold')
                title_fig.text(0.5, 0.65, f"Generated: {datetime.now().isoformat(timespec='seconds')}", ha='center', va='center', fontsize=10)
                plt.axis('off')
                self._save_fixed_pdf_page(pdf, title_fig)
                if not show_inline:
                    plt.close(title_fig)
                
                for section in ordered_sections:
                    effective_show_inline = section.show_inline_override if section.show_inline_override is not None else show_inline
                    effective_save_pdf = section.include_in_pdf
                    effective_save_images = section.include_images and save_images
                    self.render_section(section, pdf=pdf, show_inline=effective_show_inline, save_pdf=effective_save_pdf, save_image=effective_save_images)
       
        else:
            for section in ordered_sections:
                effective_show_inline = section.show_inline_override if section.show_inline_override is not None else show_inline
                effective_save_images = section.include_images and save_images
                self.render_section(section, show_inline=effective_show_inline, save_pdf=False, save_image=effective_save_images)

        if save_summary:
            self.artefacts.save_summary({
                "title": title,
                "pdf_path": str(self.artefacts.pdf_path) if save_pdf else None,
                "num_sections": len(sections),
                "sections": [{"idx": s.idx, "slug": s.slug, "title": s.title} for s in sections],
            })