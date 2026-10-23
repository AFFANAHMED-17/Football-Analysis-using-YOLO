# ⚽ Football Vision Lab

### Turn match footage into football intelligence.

**Detect. Track. Classify. Map the game.**

A computer-vision football analysis pipeline powered by custom-trained
YOLO models, player tracking, team classification, ball tracking, and a
tactical pitch radar.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Object%20Detection-YOLO-111827)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/Video%20Processing-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Project-Experimental-orange)
:::

------------------------------------------------------------------------

## 🏟️ What is this?

**Football Vision Lab** takes a football video as input and turns it
into an annotated analysis video. Instead of watching every frame with
only your eyes, the pipeline applies computer vision to identify
players, follow their movement, distinguish teams, detect the ball, and
project player positions onto a simplified top-down pitch radar.

The goal is to build a practical foundation for football analytics:
start with raw footage, run the trained models, and bring several visual
signals together in one output.

> **One video in. One annotated video out. More ways to read the game.**

## ✨ Features

  ---------------------------------------------------------------------
  Feature                            What it does
  ---------------------------------- ----------------------------------
  🧍 **Player detection**            Uses a custom-trained YOLO model
                                     to locate players in each frame.

  🏃 **Player tracking**             Associates detections across
                                     frames and displays tracking IDs
                                     when the tracker can maintain an
                                     identity.

  🎽 **Team classification**         Uses visual features from player
                                     crops to group players by team
                                     appearance.

  ⚽ **Ball detection and tracking** Runs a dedicated YOLO model for
                                     the ball and uses a tracking
                                     component to help follow it over
                                     time.

  📐 **Pitch keypoints**             Detects pitch landmarks that help
                                     relate image positions to the
                                     field.

  🗺️ **Tactical radar** U            ses a perspective transformation
                                     to map player positions onto a
                                     top-down pitch visualization.

  📊 **Progress monitoring**         Displays total video frames and a
                                     progress bar during processing.

  🎬 **Single output pipeline**      Combines the analysis overlays
                                     into one rendered output video.
  ---------------------------------------------------------------------

*Detection and tracking quality depends on the training data, camera
angle, lighting, occlusion, frame quality, and model confidence. This is
an experimental analytics project, not an official match-data provider.*

## 🧠 How it works

``` text
                 ┌──────────────────────┐
                 │  Input football video│
                 └──────────┬───────────┘
                            │
                 Read frames from video
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Player YOLO │   │  Pitch YOLO │   │   Ball YOLO │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                 │                 │
          ▼                 ▼                 ▼
   Player detections   Pitch keypoints    Ball detections
          │                 │                 │
          ▼                 │                 ▼
   Player tracking          │             Ball tracking
          │                 │                 │
          ▼                 │                 │
   Team classification      │                 │
          └─────────────────┼─────────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Annotate video +  │
                  │ tactical pitch    │
                  │ radar overlay     │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ Annotated output  │
                  │       video       │
                  └───────────────────┘
```

### Processing flow

1.  Load the trained player, pitch, and ball model weights.
2.  Read the input video and inspect its frames.
3.  Detect players and update the player tracker.
4.  Use player crops to fit the team-classification model.
5.  Detect pitch keypoints and estimate the mapping between the camera
    view and pitch coordinates.
6.  Detect the ball and update its tracking component.
7.  Draw detections, labels, tracking information, and the radar overlay
    where the available keypoints support the transformation.
8.  Write the annotated frames to the output video while showing
    progress.

The models are called as part of one pipeline. This does **not** mean
all three models necessarily execute simultaneously on separate GPU
workers; inference is coordinated frame by frame.

## 📚 The biggest part of this project: learning

This project is about more than getting a model to draw boxes around
players. **A huge part of the journey was learning how the pieces fit
together.**

We took inspiration from the football computer-vision work, public model
examples, repositories, and learning resources available through the
community---including Roboflow resources and GitHub projects. We studied
ideas and workflows, experimented with them, and used that learning to
build our own combined pipeline around our trained models and project
goals.

That learning process included understanding how object detection
differs from tracking, why tracking IDs can change or drift, how team
classification can use player crops, how pitch keypoints help map image
coordinates to a top-down radar, and how multiple models can be brought
together to process one video.

We also learned that **a working demo is not the same as a perfect
analytics system**. Models can mistake camera operators or corner flags
for players, miss a small or blurred ball, or produce inaccurate pitch
keypoints. Those errors helped us see how much dataset quality, class
definitions, model training, confidence thresholds, camera perspective,
and evaluation matter in a real computer-vision workflow.

The project is therefore both a technical build and a learning exercise:
explore existing ideas, understand the underlying methods, test
assumptions, troubleshoot compatibility issues, and combine the pieces
into a pipeline we can keep improving.

**Our approach:** learn from the open-source community, give credit
where it is due, experiment responsibly, and be transparent about what
we built and what still needs work.

### 🙏 Inspiration and attribution

Roboflow resources and public GitHub repositories were part of our
learning and inspiration. This project should not be interpreted as
claiming ownership of those original ideas, libraries, or third-party
model implementations. Please consult the specific resources that
informed your implementation, add direct links to them here, and follow
their licenses and attribution requirements. The exact references should
be documented based on the repositories and resources actually used.

## 🧰 Tech stack

-   **Python** --- pipeline and application logic
-   **Ultralytics YOLO** --- object and pitch-keypoint detection
-   **PyTorch** --- deep-learning runtime
-   **Supervision** --- detections, tracking, annotations, and video
    utilities
-   **OpenCV** --- video reading, display, and processing
-   **Transformers** --- visual feature extraction for team
    classification
-   **NumPy** --- numerical operations and coordinate transformations
-   **tqdm** --- terminal progress bar

## 📁 Project structure

Your local project should follow this general structure:

``` text
Football-Analysis/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── player-detection.pt
│   ├── field-detection.pt
│   └── ball-detection.pt
├── sports/
│   ├── annotators/
│   ├── common/
│   └── configs/
├── test-video.mp4
└── output-video.mp4
```

The exact contents of `sports/` depend on the supporting code included
in your local project. The three `.pt` files are the trained model
weights expected by the pipeline.

## 🚀 Getting started

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username:

``` bash
git clone https://github.com/YOUR-USERNAME/Football-Analysis.git
cd Football-Analysis
```

If your repository has a different name, use its actual URL and folder
name.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .ball
.\.ball\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python
directly:

``` powershell
.\.ball\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**

``` bash
python3 -m venv .ball
source .ball/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### GPU / CUDA note

For NVIDIA GPU acceleration, install a PyTorch build compatible with
your CUDA environment using the official [PyTorch installation
selector](https://pytorch.org/get-started/locally/). If you already have
a working CUDA-enabled PyTorch environment, avoid replacing it
unnecessarily.

Check GPU availability:

``` bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

If CUDA is unavailable, check your driver, PyTorch build, and
environment before running with `--device=0`. CPU inference may work but
can be much slower.

### 4. Add the model weights

Make sure these files exist in the `models/` directory:

``` text
models/player-detection.pt
models/field-detection.pt
models/ball-detection.pt
```

The weights are not automatically downloaded by this project. They must
be provided separately if they are not included in your clone. If you
publish them, confirm their size and licensing first.

### 5. Add an input video

Place a supported video file in the project directory, for example:

``` text
test-video.mp4
```

You can use another video path by changing the command below.

## 🎮 Run the full analysis

From the project root, with your environment activated:

``` powershell
python main.py --source_video_path="test-video.mp4" --target_video_path="output-video.mp4" --device=0 --mode=ALL
```

This runs the combined pipeline with the three model weights and
produces `output-video.mp4`.

If the script's default mode is configured as `ALL`, you may omit the
mode argument:

``` powershell
python main.py --source_video_path="test-video.mp4" --target_video_path="output-video.mp4" --device=0
```

### What do the arguments mean?

  ---------------------------------------------------------------------
  Argument                           Meaning
  ---------------------------------- ----------------------------------
  `--source_video_path`              Path to the input video.

  `--target_video_path`              Path where the annotated video
                                     will be saved.

  `--device=0`                       Use the first CUDA GPU, if
                                     supported by your setup.

  `--mode=ALL`                       Run the combined analysis
                                     pipeline.
  ---------------------------------------------------------------------

For CPU inference, use `--device=cpu` instead of `--device=0`.

During processing, the terminal progress bar reports the current frame,
total frames, elapsed time, and estimated remaining time when available.
A preview window may also appear if enabled by the script. Press **Q**
to stop the preview loop; stopping early can leave an incomplete output
video.

## 🎥 Output

The generated `output-video.mp4` is intended to combine:

-   Player detections and tracking labels
-   Team-based annotations
-   Ball detection and tracking visualization
-   Pitch keypoint information used by the pipeline
-   A top-down pitch radar overlay when enough valid keypoints are
    detected

The exact overlay visible on each frame depends on the detections and
keypoints returned by the models. A radar may not be reliable in frames
where pitch landmarks are missing or incorrectly detected.

## 🛠️ Troubleshooting

### `Invalid device string: '0'`

Use the updated script that converts numeric device values to CUDA
device names, or run with `--device=cpu` while diagnosing the
environment.

### CUDA is not available

Check the installed PyTorch build and NVIDIA driver:

``` bash
python -c "import torch; print(torch.cuda.is_available())"
```

Install a compatible PyTorch build if necessary.

### A dependency import fails

Confirm the virtual environment is active, then install the project's
dependencies:

``` bash
python -m pip install -r requirements.txt
```

If a specific package version is incompatible with your code, resolve
that version conflict rather than blindly upgrading every package.

### A model file cannot be found

Confirm the expected `.pt` files exist in `models/` and that the
filenames match the paths used in `main.py`.

### The ball or players are missed, or false detections appear

Possible causes include small objects, motion blur, occlusion, low
resolution, similar-looking objects, and gaps in the training dataset.
Review the model's validation results and representative frames before
changing tracking settings. A tracker can maintain IDs only when the
detections and motion estimates are good enough.

### The radar looks wrong

The radar depends on accurate pitch keypoints and a suitable
camera-to-pitch transformation. Inspect the pitch model's keypoints
first; inaccurate landmarks can lead to inaccurate projected positions.

## 🧪 Current scope and future ideas

This project focuses on the combined detection, tracking,
team-classification, and radar workflow. Potential next steps include:

-   📍 Player heatmaps and movement trails
-   🏃 Speed and distance estimation with camera calibration
-   ⚽ Ball trajectory and possession estimation
-   🔁 Pass detection and passing networks
-   🧠 Improved tracking through occlusions and camera cuts
-   📊 Team shape and formation summaries
-   🎯 Event detection for shots, corners, and goal attempts
-   📈 Match statistics and a dashboard for reviewing results

These are future enhancements, not claims about features already
implemented.

## 🤝 Contributing

Ideas, bug reports, and improvements are welcome.

1.  Fork the repository.
2.  Create a feature branch.
3.  Make a focused change and test it with a short video.
4.  Open a pull request explaining the change and its results.

When reporting a bug, include your Python version, relevant package
versions, the command you ran, and the complete error message. Avoid
uploading private footage, credentials, or very large files to an issue.

## ⚖️ Responsible use

Use video footage and trained model weights only when you have the
necessary rights and permissions. This project is intended for
experimentation and analytics development. Its detections and tracking
results can be wrong and should not be treated as authoritative referee
decisions or official performance statistics.

## 🙌 Acknowledgements

Built with the open-source Python computer-vision and deep-learning
ecosystem, including Ultralytics, PyTorch, OpenCV, Supervision, and
Transformers. Please follow each project's license and attribution
requirements.

------------------------------------------------------------------------


### Made for people who see more than just the ball. ⚽

**Detect the action. Follow the movement. Read the pitch.**

