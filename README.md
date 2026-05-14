# Chisel Data Pipeline

## Project overview
This repository is a notebook-first dataset preparation workflow for semantic segmentation. It is designed to:

- inspect a raw image/mask dataset with EDA before processing
- extract one target class at a time into a tiled dataset
- convert masks into YOLO polygon labels
- stage results in a quarantine area before merging them into the main output dataset

The current implementation is centered on `notebooks/template.ipynb`, `pipeline.py`, and `eda.py`. 

Target classes are defined in `config.py`:

| `class_id` | class name |
| --- | --- |
| `1` | `building` |
| `2` | `vegetation` |
| `3` | `water` |
| `4` | `road` |

Each run produces a single-class dataset. Even when the source masks are multiclass, generated YOLO labels are written with class `0` and `data.yaml` is created with `nc: 1`.

## Repository structure
```text
.
├── config.py                  # shared config and class map
├── pipeline.py                # main processing entrypoint
├── eda.py                     # dataset EDA and report generation
├── data/                      # Natural Earth country lookup assets used by spatial metadata code
├── libs/
│   ├── annotation_generator.py
│   ├── file_handler.py
│   ├── image_processor.py
│   ├── mask_filters.py
│   ├── spatial_metadata.py
│   ├── utils.py
│   ├── visualiser.py
│   ├── write_report.py
│   └── eda/
├── models/
│   ├── context.py
│   ├── metadata.py
│   └── reportsection.py
└── notebooks/
    ├── template.ipynb         # primary user workflow
    ├── quarantine/            # staged processed datasets
    ├── output/                # merged datasets
    └── reports/               # EDA reports and summaries
```

Raw datasets are not stored in this repository. The notebook expects `data_dir` to point to an external dataset directory on your machine.

## Installation
Dependencies are managed with Poetry. 

The project currently targets Python `3.11+`. If you do not have `3.11`, use [Pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions and switch to `3.11` before set-up. You can skip the poetry installation step if you already have it.

```bash
python --version    # ensure version is 3.11
pip install poetry
poetry env use python3.11
poetry install --with notebook
source $(poetry env info --path)/bin/activate
python -m ipykernel install --user --name chisel-pipeline-py3.11 --display-name "Chisel pipeline (py3.11)"
```

Notes:

- `poetry install` installs the runtime dependencies from `pyproject.toml` using the locked versions in `poetry.lock`.
- `--with notebook` also installs the notebook tooling declared under the `notebook` dependency group, including `jupyterlab` and `ipykernel`.
- `kaleido` is needed for Plotly image export in the EDA report flow.
- Geospatial packages such as `rasterio` and `geopandas` may require system libraries depending on your environment.
- The last two steps 

## Environment setup:
If all of the above steps have been completed correctly, when you open `template.ipynb` you should be able to select the poetry environment in the kernel, in the top right. 

1. Keep your raw dataset outside this repo and note its path.
2. Verify that the Natural Earth country lookup files expected by `libs/spatial_metadata.py` are available under `data/`.

## Data/input formats
`pipeline.py` loads a dataset by recursively scanning `data_dir` and pairing images with masks using `libs/file_handler.py`.

Current pairing rules:

- image and mask files are matched by normalised filename stem
- files inside directories named `mask`, `masks`, `label`, or `labels` are treated as masks
- filenames with prefixes `mask_` or `label_` are treated as masks
- filenames with suffixes `_mask` or `_m` are treated as masks
- ambiguous matches or missing image/mask pairs are skipped and logged rather than hard-failing

Supported source handling:

- `.tif` and `.tiff` are read with Rasterio
- other image formats are read with OpenCV
- masks are expected to be image files, not YOLO `.txt` files

Mask processing behavior:

- binary masks with values like `0/1` or `0/255` are normalised automatically
- dataset-specific multiclass filters currently exist for `chesapeake`, `dubai`, `landcover`, and `openearthmap-full`
- unknown indexed multiclass masks fall back to a generic class filter
- unknown 3-channel color masks need a dataset-specific filter in `libs/mask_filters.py`

It is recommended that you write a dataset-specific filter for your multi-class datasets based on what you learn from the EDA pre-processing report. 

## Running the project
The intended execution path is `notebooks/template.ipynb`. The notebook runs the pipeline in this order:

1. import project modules
2. set `data_dir`, `set_name`, and `class_id`
3. optionally define manual `SpatialMetadata`
4. run pre-processing EDA with `DatasetEDA(..., stage="pre")`
5. run `process_dataset_incremental(...)`
6. run `PipelineUtils(...).clean_dataset(...)`
7. run post-processing EDA on the quarantined class folder
8. merge the quarantined set into the main output dataset with `FileHandler.merge_quarantine_sets(...)`

Example parameter cell from the template:

```python
data_dir = "../../../data/OpenEarthMap_final"
set_name = "openearthmap-full"
class_id = 1
class_name = CLASS_ID_MAP[class_id]
```

## Scripts and utilities
Important modules in the current codebase:

- `pipeline.py`: main processing function `process_dataset_incremental(data_dir, set_name, class_id, config)`
- `eda.py`: `DatasetEDA`, which builds pre/post reports and writes PDFs, summary JSON, and section images
- `libs/file_handler.py`: dataset discovery, image/mask pairing, YOLO label writing, and quarantine merge utilities
- `libs/image_processor.py`: image loading, tiling, mask normalization, and dataset-specific mask dispatch
- `libs/mask_filters.py`: per-dataset mask conversion to a binary target class
- `libs/annotation_generator.py`: contour extraction and YOLO polygon generation
- `libs/spatial_metadata.py`: CRS, bounds, resolution, centroid, and country metadata extraction
- `libs/utils.py`: class folder creation, `data.yaml` writing, CSV tracking, and cleanup passes
- `libs/visualiser.py`: utilities for checking image/mask/label triplets and overlaying annotations

## API usage
There is no HTTP API or CLI wrapper in the current repo. The callable API is the Python module surface used by the notebook.

Example programmatic flow:

```python
import os

from config import config, CLASS_ID_MAP
from eda import DatasetEDA
from libs.file_handler import FileHandler
from libs.utils import PipelineUtils
from pipeline import process_dataset_incremental

data_dir = "../../../data/OpenEarthMap_final"
set_name = "openearthmap-full"
class_id = 1
class_name = CLASS_ID_MAP[class_id]

eda_pre = DatasetEDA(data_dir, class_id, set_name, config, stage="pre")
eda_pre.report(show_inline=False, save_pdf=True, save_images=True, save_summary=True, output_dir="reports")

process_dataset_incremental(data_dir=data_dir, set_name=set_name, class_id=class_id, config=config)

set_dir = os.path.join(config["quarantine_dir"], set_name)
PipelineUtils(config, set_dir).clean_dataset(class_name)

class_dir = os.path.join(set_dir, class_name)
eda_post = DatasetEDA(class_dir, class_id, set_name, config, stage="post")
eda_post.report(show_inline=False, save_pdf=True, save_images=True, save_summary=True, output_dir="reports")

FileHandler(config).merge_quarantine_sets(set_dir, False, True)
```

## Example workflow
Use this flow for a new dataset:

1. Launch `notebooks/template.ipynb` from the `notebooks/` directory.
2. Set `data_dir` to the raw dataset root.
3. Pick a short `set_name`. This string is not just metadata; it also controls dataset-specific mask filtering.
4. Set `class_id` for the target class you want to export.
5. Run pre-EDA and review the generated report before processing. Target `class_id` has no impact on this stage.
6. Run the processing cell to create tiled images, masks, labels, CSV rows, and `data.yaml` in quarantine.
7. Run the cleanup cell to remove degenerate labels and unlabeled tiles.
8. Run post-EDA on the quarantined class directory.
9. Merge the quarantined set only after you are satisfied with the outputs.

## Outputs/artifacts
When launched from `notebooks/`, the pipeline produces:

```text
notebooks/
├── quarantine/
│   ├── pipeline.log
│   └── <set_name>/
│       └── <class_name>/
│           ├── data.yaml
│           ├── dataset_file_directory.csv
│           ├── images/
│           ├── masks/
│           └── labels/
├── output/
│   └── <class_name>/
│       ├── data.yaml
│       ├── dataset_file_directory.csv
│       ├── images/
│       ├── masks/
│       └── labels/
└── reports/
    └── <set_name>/
        └── <timestamp>_{pre|post}/
            ├── report_<set_name>_<stage>-processing.pdf
            ├── summary.json
            └── images/
```

Key artifact details:

- `labels/*.txt` are YOLO polygon annotations, not mask images
- `dataset_file_directory.csv` tracks source filenames, deletion status, multiclass flag, spatial metadata, and country lookup results
- `data.yaml` is written as a single-class dataset config

## Development notes
Current extension points for new datasets:

- add a new mask filter to `libs/mask_filters.py`
- wire that filter into `ImageProcessor.filter_mask_by_class_id()`
- keep filter output as a 2D `uint8` mask with values `0` and `255`

Other useful implementation details:

- tiling size is controlled by `config["target_size"]` and currently defaults to `512`
- `process_dataset_incremental()` writes CSV rows for all tiles, including skipped tiles marked as `deletion_status="deleted"`
- `clean_dataset()` performs a second cleanup pass after processing by removing degenerate label files and unlabeled image/mask pairs
- `delete_set_from_class()` allows you to remove a set from one class if you change your mind. 

## Known limitations
- Output paths are working-directory sensitive because `config.py` uses `os.getcwd()`.
- The pipeline is single-class per run. YOLO labels are always rewritten as class `0`.
- `data.yaml` points both `train` and `val` to `images/`; the split percentages are metadata, not separate directory trees.
- `set_name` is part of the processing logic. A typo can change which mask filter is applied.
- Spatial metadata and country lookup depend on Natural Earth files under `data/`. Verify the full shapefile set is present if you rely on that path.
- Plotly static export depends on `kaleido`.
- EDA can be memory-heavy on larger datasets.

## Future Work 
- The final form of this pipeline will include training. Ideally, this should allow users to:
    - select specific subsets of data for training, whether by dataset name or country, or some other value that is logged in the csv
    - use analysis functions from the EDA profiler to add additional layers of information (e.g., the use of the different vegetation indices methods)
    - track all of these and other training metrics (time taken, epochs, pre-processing methods, metrics, etc.)
    - do model versioning

## Contact Details
Please contact me at chen.wynne@gmail.com if you have any issues. 
