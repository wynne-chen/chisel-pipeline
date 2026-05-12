from models.reportsection import ReportSection


class EDASections:
    '''
    Section orchestration extracted from DatasetEDA.
    '''

    def __init__(self, eda):
        self.eda = eda

    def build_sections(self, context: dict) -> list[ReportSection]:
        sections: list[ReportSection] = []
        sections.append(self.eda._build_section_file_types(idx=1))
        sections.append(self.eda._build_section_unique_values(idx=2))
        sections.append(self.eda._build_section_random_grid(idx=3))
        sections.append(self.eda._build_section_spatial_metadata(context=context, idx=4))
        sections.append(self.eda._build_section_basic_metrics(idx=5))
        sections.append(self.eda._build_section_image_size_distribution(idx=6))
        sections.append(self.eda._build_section_colour_distribution_analysis(idx=7))
        sections.append(self.eda._build_section_intensity_analysis(idx=8))
        sections.append(self.eda._build_section_average_image(context=context, idx=9))
        sections.append(self.eda._build_section_intensity_heatmap(context=context, idx=10))
        sections.append(self.eda._build_section_contrast_heatmap(context=context, idx=11))
        sections.append(self.eda._build_section_vegetation_water_indices(context=context, idx=12))

        self.eda._compute_multiclass_summary(context)
        if context["foreground_classes"]:
            for class_id in context["foreground_classes"]:
                sections.append(
                    self.eda._build_section_object_statistics_per_class(
                        context=context,
                        idx=13,
                        class_id=class_id,
                        include_in_pdf=True,
                        include_images=True,
                        include_text=True,
                        show_inline_override=None,
                    )
                )
        else:
            sections.append(self.eda._build_section_object_statistics_per_class(context=context, idx=13, class_id=self.eda.class_id))

        if not self.eda.only_one_class:
            sections.append(self.eda._build_section_multiclass_object_summary(context=context, idx=14))

        if self.eda.stage == 'post':
            sections.append(self.eda._build_section_triplet_check(context=context, idx=15))
        return sections
