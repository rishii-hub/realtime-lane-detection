# Configuration

Every tunable parameter lives in typed dataclasses (`app/config.py`) and can be
overridden with a YAML file. Nothing is a hidden constant.

## Loading a config

```python
from app.config import PipelineConfig

# Defaults
config = PipelineConfig()

# From YAML
config = PipelineConfig.from_yaml("configs/default.yaml")
```

From the CLI:

```bash
python -m app --source samples/highway_drive.mp4 --config configs/default.yaml
```

## Reference

### `camera`

| Key                    | Default | Description                                        |
| ---------------------- | ------- | -------------------------------------------------- |
| `width`                | `640`   | Requested capture width (px)                       |
| `height`               | `480`   | Requested capture height (px)                      |
| `fps`                  | `30`    | Requested capture frame rate                       |
| `max_processing_width` | `800`   | Frames wider than this are downscaled for speed    |

### `detection`

| Key                   | Default | Description                                                 |
| --------------------- | ------- | ----------------------------------------------------------- |
| `gaussian_kernel`     | `5`     | Blur kernel size (must be **odd**)                          |
| `clahe_clip_limit`    | `2.0`   | CLAHE contrast clip limit                                   |
| `clahe_grid_size`     | `8`     | CLAHE tile grid size                                        |
| `canny_low`           | `50`    | Lower Canny hysteresis threshold                            |
| `canny_high`          | `150`   | Upper Canny hysteresis threshold                            |
| `roi_bottom_width`    | `0.90`  | ROI base width as a fraction of frame width                 |
| `roi_top_width`       | `0.20`  | ROI top width as a fraction of frame width                  |
| `roi_horizon`         | `0.60`  | Vertical position of the ROI top (fraction of height)       |
| `hough_rho`           | `2`     | Distance resolution of the Hough accumulator (px)           |
| `hough_threshold`     | `40`    | Minimum votes for a detected line                           |
| `hough_min_line_length` | `30`  | Minimum segment length (px)                                 |
| `hough_max_line_gap`  | `100`   | Maximum gap to bridge collinear segments (px)               |
| `min_slope`           | `0.4`   | Reject segments flatter than this                           |
| `max_slope`           | `2.5`   | Reject segments steeper than this                           |
| `smoothing_window`    | `5`     | Number of frames in the temporal-smoothing buffer           |

### `visualization`

| Key                      | Default | Description                                            |
| ------------------------ | ------- | ------------------------------------------------------ |
| `lane_thickness`         | `10`    | Lane line thickness (px)                               |
| `fill_alpha`             | `0.30`  | Opacity of the filled lane region                      |
| `deviation_threshold_px` | `50`    | Offset beyond which the HUD warns                      |
| `show_hud`               | `true`  | Toggle the FPS/latency overlay                         |

## Validation

Invalid values fail fast at construction time. For example:

```python
DetectionConfig(gaussian_kernel=4)   # ValueError: gaussian_kernel must be odd
DetectionConfig(canny_low=200, canny_high=100)  # ValueError: low >= high
CameraConfig(width=0)                # ValueError: dimensions must be positive
```

## Tuning tips

- **Missing faint lanes?** Raise `clahe_clip_limit`, lower `canny_low`.
- **Too much noise / phantom lanes?** Raise `hough_threshold` and `min_slope`.
- **Overlay jitters?** Increase `smoothing_window` (at the cost of responsiveness).
- **Camera mounted high/low?** Adjust `roi_horizon` and the ROI widths.
