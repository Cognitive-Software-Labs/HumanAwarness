# Design doc: full pipeline reference

## What this is

A full transcription (translated from the original Romanian, tables kept
intact) of `~/Downloads/Pipeline awareness robot2.pdf` ("Pipeline de
percepție pentru cooperarea om-robot" / "Human awareness detection
pipeline") — the project's design presentation, referenced throughout
`docs/context/*.md` as "the design doc" but never itself checked into the
repo. That PDF only exists on one machine; this file is the durable,
version-controlled copy of its actual content, so "see the design doc"
means something to a reader who doesn't have that Downloads folder.

**This describes the target design, not current state.** Only a slice of
it is built - `human_perception` + `human_tracking` (this doc's "A ·
Detecție + tracking"), `human_head_pose` ("B1 · Orientarea capului"), and
`human_gaze` ("B2 · Iris și privire", both phases), and the awareness
dimension of the per-ID aggregation ("D · Agregare"; `human_awareness`).
The environment stage ("C · Mediu") and the rest of D (availability,
willingness, the combined cooperation score) are not started. Where a built
part's own context file (`bbox_keypoints.md`, `human_tracking.md`,
`head_pose.md`, `gaze.md`, `awareness.md`) already documents the actual
implementation decisions and
real-footage validation, this file isn't a replacement for that - it's
the earlier, broader survey those decisions were drawn from (the full set
of candidates considered, not just the one picked).

## 00 · Conceptual diagram

Cooperation isn't visible in a single frame - it's inferred from behavior
over time. Each finer level only runs when resolution allows; otherwise
the state stays "unknown," not negative.

Built on ROS2, a monocular RGB camera, ~10-15 fps, producing a
cooperation score per person:

1. **Person detection** - bbox + skeleton (keypoints)
2. **Stable-ID tracking** - accumulated state per ID
3. **Geometry** - radius, distance, position in odom
4. **Cues, gated on pixel size** - torso (skeleton visible), head (face ≥
   40px), gaze/iris (eyes ≥ 20px)
5. **Per-ID state, with decay** - awareness → availability → willingness,
   each with a value + confidence
6. **Environment** (async, ~1 Hz) - free space, density, doors, queues;
   modulates interpretation
7. **Cooperation score** - percentage + confidence + reason

## Part I — Models considered, concept by concept

For each step: the candidates, their pros and cons, and the decision. The
highlighted row (**bold decision**) is the model actually chosen.

### 1 · Person detection + skeleton

Pose detector: one-stage or top-down. Gives bbox + keypoints every frame.
Since distant people don't matter here, RTMPose's enlarged-crop advantage
disappears.

| Model | Pro | Con | Decision |
|---|---|---|---|
| **YOLO26-pose (s / m)** | One model for bbox + 17 keypoints; s variant: 63.0 mAP in 2.7ms (T4, TensorRT); tested on Jetson; active community | AGPL-3.0 license; image downscaled to 640px, coarse skeleton at distance; only 17 body points | **Chosen (research)** |
| RTMPose top-down (+RTMDet) | Apache 2.0; RTMPose-m 75.8 AP on COCO; enlarged crop helps at distance; RTMW has face + hands; rtmlib with no mmcv | Two models; cost grows with number of people; MMPose unmaintained since 2024; TensorRT can hurt accuracy | Commercial alternative |
| RTMO (one-stage) | RTMO-l 74.8 AP; 73.2 AP on CrowdPose; near-constant time regardless of people count | Loses the enlarged-crop advantage; same weak maintenance as MMPose | Option |

### 2 · Tracking with a stable ID

Trackers: what matters is IDF1, not MOTA. A changed ID loses all
accumulated awareness. The camera moves with the robot, so camera-motion
compensation is mandatory.

| Model | Pro | Con | Decision |
|---|---|---|---|
| ByteTrack | Two-stage association recovers weak detections; ~43% of one CPU core; mature | No ReID; no camera-motion compensation | Baseline |
| OC-SORT | Corrects Kalman after occlusion; fast, no ReID | No visual appearance: new ID after a few seconds; only helps with erratic motion | Rejected |
| **BoT-SORT + ReID** | CMC for a moving camera; HOTA 63.7 / IDF1 78.7 on MOT17; default in Ultralytics | ~30% slower than ByteTrack; image-based CMC fails in crowds | **Chosen** |
| BoostTrack, StrongSORT, Deep OC-SORT | Best scores: BoostTrack HOTA 69.3 / IDF1 83.2 | 12-25 FPS in Python on desktop; many hyperparameters; BoxMOT is AGPL | Upgrade |
| MOTIP | End-to-end, ID prediction; HOTA 73.7 on DanceTrack | Below BoT-SORT on pedestrians (59.2); 13-23 FPS on server GPU; black box | Rejected |
| SAM 3 / 3.1 | Pixel masks, text prompt "person," single model | ~30 FPS only at ~5 objects, on H100/H200; loses identity across occlusions | Offline only |

### 2b · Re-identification (ReID)

The component that keeps the ID stable: turns a person crop into an
appearance vector (clothes, colors, shapes). Helps after occlusion or
after the person leaves and re-enters frame.

| Model | Pro | Con | Decision |
|---|---|---|---|
| **OSNet (x0.25 / x1.0)** | Very small, fast on Jetson; 512-number vector; standard in BoxMOT | Medium discrimination between similar clothing | **Chosen** |
| CLIP-ReID | More robust to light and angle | Heavier | Upgrade |
| KPR (ECCV 2024) | Uses existing keypoints; separates 2 people in the same bbox | Heavier; lightly tested in production | If overlaps appear |

**ID lifecycle**: `NEW` → 3 frames → `ACTIVE` → `LOST` (Kalman + ReID, 2-3s)
→ `IN GALLERY` (ReID only, 30-60s) → `FORGOTTEN`. A new person is first
compared against the gallery; if it matches, it inherits the old ID and
its awareness state.

### 3 · Head orientation

Head pose: needs to see the head, not the face. "Facing away" is real
information - a model trained only on ±90° yaw gives a plausible-looking
but false reading at the back, so a full-range (360°) model is required.

| Model | Pro | Con | Decision |
|---|---|---|---|
| **6DRepNet360** | Yaw + pitch + roll over 360°; 6D rotation, no jump at ±180°; IEEE TIP 2024; ONNX from PINTO | Licensing is awkward for commercial use; doesn't give confidence; weaker from behind | **Chosen** |
| YawNet (PINTO) | Yaw 360° (cos/sin), distilled from DINOv3; synthetic data with blur/occlusion; clean license | Yaw only; new project, needs validating on real footage | Commercial alternative |
| WHENet | First fine-grained full-range model; compact, mobile-oriented | Older, less accurate, dated stack | Eliminated |
| DirectMHP | Head detection + pose for multiple people in one model | Duplicates existing detection; heavy; restrictive licenses | Eliminated |
| HopeNet, 6DRepNet, FSA-Net | Mature, well documented | Trained only on frontal faces (±90°): false result from behind | Eliminated |

### 4 · Iris and gaze

Gaze: where the eyes point, not the head. Gaze = head orientation + eye
rotation (up to ~30°). Only runs up close, when the eye is ≥ 20-30px.

| Model | Pro | Con | Decision |
|---|---|---|---|
| **MediaPipe Face Landmarker** | 478 points including iris; blink, gating, crop; light and interpretable | Doesn't estimate gaze itself; its geometric approximation has 5-10° error | **Chosen (gating)** |
| **L2CS-Net / MobileGaze (Gaze360)** | Face crop → gaze directly in the camera frame; real-time; covers large angles | Sensitive to framing; gives an angle even with closed eyes; non-commercial-licensed weights | **Chosen (gaze)** |
| GaT (gaze3d, CVPR 2025) | On video; best for moving people and large angles | Swin3D is heavy: hard at 10fps for multiple people | Upgrade with GPU |
| UniGaze / 3DGazeNet | Very good generalization to new people/conditions | Large ViT; lightly tested in production | Rejected |
| Eye Contact CNN (Chong 2020) | Directly answers "looking at the camera?" with a probability | Trained on adult-child interactions; yes/no only; likely needs retraining | Option |
| Gaze-LLE (CVPR 2025) | Predicts where in the image a person is looking | The robot *is* the camera, so it never appears in-frame | Not a fit |

### 5 · Environment, async

Environment: named concepts, not an opaque label. The environment doesn't
produce a score, it changes interpretation. Density, groups, and free
space don't need a new model - they come from tracks, head pose, and the
costmap.

| Model | Pro | Con | Decision |
|---|---|---|---|
| **Label-free CBM (simplified)** | Concepts placed automatically by the encoder; only class labels needed; a new concept is just a sentence | Zero-shot scores aren't probabilities; leakage risk; the "tail" is fragile | **Chosen** |
| CBM classic (Koh 2020) | Explainable decision via concepts | Needs manual annotation for every concept | Rejected |
| **SigLIP 2 (encoder)** | Sigmoid: independent score per concept; beats SigLIP zero-shot | Trained on eye-level photos; a low camera reduces accuracy | **Chosen** |
| MobileCLIP2 | Small, fits limited hardware | Lower accuracy than SigLIP 2 | Reserve |
| YOLOE | Open-vocabulary detection via text prompt, real-time, gives position | AGPL-3.0 license; one more model on the robot | Eliminated |
| YOLO-World | Open-vocabulary alternative | Older; GPL-v3 license | Eliminated |

## Part II — Final pipeline: models, components, adapters

Models are black boxes. Components are your own code: crop, gating,
geometry, time filters. Adapters translate the data structure between
them. Legend: **Data** · **AI model (black box)** · *Component (own
code)* · → *adapter*.

### Pipeline overview

**A · Main loop, every frame (~10-15 fps)** - all detections enter tracking:

`RGB frame + odom (640px)` → `YOLO26-pose (bbox + 17kp)` → `BoT-SORT +
OSNet (CMC from odom)` → `ReID gallery (30-60s)` → `stable ID (bbox + kp
per ID)`

**B · Per ID, cascaded** - each model only runs above its own pixel
threshold (a parameter):

`Head crop, neck up` → `6DRepNet360 (head ≥ 35px)` → `face-to-radius
angle (Gram-Schmidt)` → `MediaPipe Landmarker (eye gating, blur, yaw)` →
`L2CS-Net gaze` → `visible cone + peripheral buffer`

**C · Environment, async (~1 Hz)**, a separate ROS2 node:

`Sampled frame (throttle 1Hz)` → `SigLIP 2 (~25 concepts)` → `sparse CBM
head (place type)` → `context topic (last valid state)`

**D · Per-ID aggregation (no new model)**:

`Geometric cues (trajectory, torso, gestures, group)` → `temporal fusion
per ID (awareness × availability × willingness, with decay; context
modulates)` → `cooperation score (% + confidence + reason)`

### Pipeline · A Detection + tracking

From a frame to a stable ID. Two models (detector and ReID) are black
boxes; ID stability comes from the components between them.

| | Input | AI model | Component |
|---|---|---|---|
| | RGB frame + odometry, 640px, ~10-15fps; robot yaw from odom | YOLO26-pose (s/m): image → bbox, 17 keypoints, score per point | CMC from odometry: robot rotation moves the Kalman predictions |
| | | **AI model + algorithm**: BoT-SORT + OSNet ReID (Kalman + IoU/appearance association; ReID only on confident detections; 2-3s buffer) | **Component**: ReID gallery, 30-60s (5-10 clean samples per ID, matched with a margin over the second-best candidate) → **Output**: stable ID per person (bbox + keypoints + confidence; state is inherited on return) |

No distance filter: every detection enters tracking; pixel thresholds
only apply per-model, in the cascade. A person exiting at the frame edge
goes straight into the gallery.

### Pipeline · B1 Head orientation

From keypoints to "looking toward the robot?" No Euler angles needed -
the angle between the face direction and the head→camera ray handles
people at the frame edge correctly too.

| | Input | Component | AI model |
|---|---|---|---|
| | Keypoints + frame: shoulders, hips, nose, eyes, ears of the ID | Head crop, neck up: neck = shoulder midpoint; square crop, side = `k_h × scale`, along the hip→neck axis; 224×224 | 6DRepNet360 (ONNX): crop → 6 numbers (6D rotation), over 360° |
| | | **Component**: Gram-Schmidt → R (3×3 matrix; face direction = R·[0,0,1]) | **Component**: angle to the ray (arccos(face·−ray); confidence from pixels, keypoint scores, consistency) → **Output**: 0°-180° angle (~0° at the robot, ~90° profile, ~180° from behind; + pitch for "on the phone?") |

Why the neck, not "30% of the bbox": the bbox varies with raised arms,
leg occlusion, or leaning; the neck stays at the head's base and still
works from behind, when the nose and eyes are missing. Below
`min_head_px`: falls back to the coarse keypoint-only level.

### Pipeline · B2 Iris and gaze

MediaPipe prepares, L2CS-Net estimates. Only runs up close. When gaze is
available it replaces head pose in the angle calculation; otherwise it
falls back to head pose.

| | Input | AI model | Component |
|---|---|---|---|
| | Head-region crop from tracking, + head pose from B1 | MediaPipe Face Landmarker: 478 3D points including iris (5 per eye) | Gating + face crop: eye ≥ 20-30px, \|yaw\| < 45°, eye open, no blur or reflections |
| | | **AI model**: L2CS-Net (Gaze360): face crop → gaze yaw/pitch, in the camera frame | **Component**: check + filter (\|gaze − head\| < 30°; angle to −ray; One Euro; fixations only) → **Output**: "Sees the robot?" (θ ≤ robot cone + peripheral buffer → yes; separately: direct contact; + confidence) |

Peripheral vision: the robot can be seen without direct eye contact with
the camera. The threshold isn't a point, it's the robot's cone,
`atan(r_robot / distance)`, plus a buffer of a few degrees (default 5°,
a parameter). A blink doesn't break a fixation.

### Pipeline · C Environment, async ~1 Hz

Label-free CBM over SigLIP 2. A separate ROS2 node, subscribed to images
with throttling. Doesn't block the main loop; aggregation reads the last
valid state.

| | Input | AI model | Component | Component |
|---|---|---|---|---|
| | Sampled frame, ~1 frame/second | SigLIP 2 encoder: image → vector; concept texts computed once | Concept scores: ~20-30 concepts, thresholds calibrated on your own data | Sparse linear head: logistic regression; only sees the concepts → place type |
| | **Component (no model)**: tracks + costmap, density, groups (F-formations), free-space width | | | **Output**: context topic (place type, active concepts, zones, density; modulates aggregation) |

Without YOLOE: doors and elevators stay frame-level concepts ("there's a
door in the image"), with no position in the world. Example of an
explainable chain: "narrow corridor" active → small deviation budget →
not deviating isn't penalized as a refusal.

### Parameters

Legend: **new parameter** · *changed value* · everything else unchanged.
All live in `params.yaml`.

#### A · Detection + tracking

| parameter | recommended | what it does |
|---|---|---|
| `det_model` | yolo26s-pose | n/s/m variant: speed vs. accuracy |
| `input_size` | 640 | image side length; larger = people further away stay usable |
| `fps_target` | 10 | loop frequency; basis for parameters expressed in seconds |
| `track_high_thresh` | 0.5 | score for the first association stage |
| `track_low_thresh` | 0.1 | below this, a detection is discarded entirely |
| `new_track_thresh` | 0.6 | minimum score to start a new ID |
| `kp_conf_min` | 0.5 | below this, a keypoint is treated as missing |
| `track_buffer_s` | 2.5 | how long a lost track lives (× fps → frames) |
| `cmc_source` | odom | camera-motion compensation: odom or image |
| `gallery_ttl_s` | 45 | how long an expired ID stays in the ReID gallery |
| `gallery_samples` | 8 | clean samples per ID, from different angles |
| `reid_min_det_conf` | 0.7 | minimum score for a sample to enter the gallery |
| `reid_dist_max` | 0.3 | maximum cosine distance for a match |
| `reid_margin` | 0.1 | minimum lead over the second-best candidate |

#### B · Head + gating

| parameter | recommended | what it does |
|---|---|---|
| `head_k_h` | 0.8 | crop height from the neck up, × scale |
| `head_scale_alpha` | 1.7 | scale = max(shoulders, α × neck-nose), for profile views |
| `crop_margin` | 0.2 | margin on each side; the only source of padding |
| `min_head_px` | 35 | below this, 6DRepNet360 doesn't run |
| `mp_min_head_px` | 80 | below this, MediaPipe doesn't run |
| `min_eye_px` | 20 | eye width; below this, L2CS-Net doesn't run |
| `max_yaw_gaze_deg` | 45 | head turned from the ray past this → no gaze attempt |
| `max_eye_head_deg` | 30 | gaze − head past this → gaze reading is discarded |
| `ear_min` | 0.2 | below this, the eye is closed (blink) → frame skipped |
| `blur_min` | calibrated | Laplacian variance over the eye region |

#### B · Gaze over time + angles

| parameter | recommended | what it does |
|---|---|---|
| `fixation_min_ms` | 300 | minimum duration of a fixation (3 frames at 10fps) |
| `fixation_max_dev_deg` | 4 | maximum angle variation during a fixation |
| `blink_gap_max_ms` | 300 | tolerated gap without breaking the fixation |
| `oneeuro_min_cutoff` | calibrated | smoothing when gaze holds still |
| `oneeuro_beta` | calibrated | how fast real movements come through |
| `theta_contact_deg` | 12 | direct contact: gaze angle to the camera ray |
| `robot_radius_m` | 0.3 | half the robot's width → cone atan(r/d) |
| `peripheral_buffer_deg` | 5 | degrees added past the robot's cone: peripheral vision |
| `head_only_extra_deg` | 15 | widens the threshold when only head pose is available |

#### Geometry + trajectory

| parameter | recommended | what it does |
|---|---|---|
| `camera_height_m` | measured | distance via the ground plane, from the ankles |
| `shoulder_width_prior_m` | 0.40 | distance from shoulder width, when legs aren't visible |
| `person_height_prior_m` | 1.70 | distance from height, as a last resort |
| `camera_frame` | from tf | the camera's optical frame |
| `robot_base_frame` | base_link | "toward the robot" angle, relative to its center |
| `traj_window_s` | 2.0 | window for speed and direction |
| `slowdown_ratio` | 0.7 | below 70% of previous speed = slowing down |
| `stop_speed_mps` | 0.15 | below this, the person is considered stopped |
| `deviation_min_m` | 0.3 | minimum lateral displacement to count as a deviation |

#### C · Environment

| parameter | recommended | what it does |
|---|---|---|
| `env_rate_hz` | 1.0 | SigLIP 2 frequency while the robot drives straight |
| `env_rate_hz_turning` | 3.0 | frequency while the robot is turning |
| `siglip_model` | so400m-384 | encoder variant and resolution |
| `concepts_file` | concepts.yaml | concepts, with multiple phrasings each |
| `concept_threshold` | p90 negative | per-concept threshold, calibrated on your own images |
| `cbm_head_file` | cbm.npz | head weights + mean/std for z-scoring |
| `env_ema_s` | 5 | smoothing of concept scores over time |
| `place_switch_s` | 3 | hysteresis on place-type changes |
| `density_radius_m` | 3 | radius for local density |
| `group_max_dist_m` | 1.5 | maximum distance between group members |
| `group_facing_deg` | 60 | mutual orientation for an F-formation |
| `narrow_width_m` | 1.5 | below this, not deviating isn't a refusal |

#### D · Aggregation

| parameter | recommended | what it does |
|---|---|---|
| `tau_awareness_s` | 8 | awareness decay: "knows you're there" |
| `tau_availability_s` | 4 | availability decay: phone, conversation |
| `tau_willingness_s` | 2 | willingness decay: current behavior |
| `fusion_weights` | weights.yaml | weight per cue and per source (iris/head/body) |
| `cue_max_age_s` | 0.5 | maximum age of a cue per frame |
| `env_max_age_s` | 3 | maximum age of the environment context |
| `init_value` | 0.5, conf 0 | a new ID starts "don't know," not "no" |
| `combine` | geometric_mean | how the three dimensions combine |
| `min_confidence` | 0.3 | below this, a dimension doesn't enter the combination |
| `score_on` | 0.6 | above this: cooperating |
| `score_off` | 0.45 | below this: no longer cooperating (hysteresis) |
| `output_rate_hz` | 10 | score publish frequency |
| `context_modulation_file` | context.yaml | context → weights rules |

## Part III — Adapters: the data contracts

For every arrow in the pipeline: what structure comes out, what structure
the next component expects, and the conversion that bridges them.

### Principle

A component's output is rarely exactly the next one's input:

`YOLO26-pose` outputs a `[1, 300, 57]` tensor in letterbox coordinates →
**adapter**: undo the letterbox, split out keypoints → `BoT-SORT` expects
`N×6` `[x1,y1,x2,y2,score,cls]` in frame pixels.

What commonly doesn't match:

| Shape | Type and range | Color order | Coordinate space |
|---|---|---|---|
| H×W×C → N×C×H×W, batched | uint8 0-255 → float32 normalized | BGR (OpenCV) → RGB (models) | letterbox, crop, 0-1 → frame pixels |

| Axis convention | Representation | Time | Transport |
|---|---|---|---|
| model axes → optical frame | bins → degrees → vector; 6D → R | timestamps, 15Hz vs 1Hz | internal object → ROS2 message |

Every component declares its own input/output contract. The adapter is a
pure function, tested on its own. Swapping a model only ever touches its
adapters.

### Adapters · A Detection + tracking

From the ROS image to a `PersonTrack` message.

| Between | Out | In (expected) | Adapter |
|---|---|---|---|
| Camera → YOLO26-pose | `sensor_msgs/Image`, bgr8, W×H | float32 `[1,3,640,640]`, RGB, 0-1 | cv_bridge, BGR→RGB, letterbox, ÷255, NCHW; keeps the scale and padding |
| YOLO26-pose → BoT-SORT | `[1,300,57]`: bbox, score, cls, 17×(x,y,c), letterbox | N×6 `[x1,y1,x2,y2,score,cls]`, frame pixels | undo letterbox, score threshold, keypoints set aside, linked by index |
| Odometry → CMC | `nav_msgs/Odometry`, quaternion, ~50Hz | 2×3 warp matrix between frames | interpolate to timestamp, Δyaw, H = K·R(Δyaw)·K⁻¹, first 2 rows |
| Tracks → OSNet | bboxes in frame pixels | float32 `[B,3,256,128]`, ImageNet-normalized | crop, resize 256×128, normalize, batch |
| OSNet → gallery | embedding `[B,512]` | unit vector linked to an ID | L2 normalize, associate with ID |
| Tracker → downstream | internal STrack objects | `PersonTrackArray`: id, bbox, kp 17×3, conf, stamp | serialize into a dedicated ROS2 message |

### Adapters · B1 Head orientation

From `PersonTrack` to a face vector.

| Between | Out | In (expected) | Adapter |
|---|---|---|---|
| PersonTrack → head crop | kp 17×(x,y,c), frame pixels | shoulders, hips, head points + visibility flag | select COCO indices, flag missing below `kp_conf_min` |
| Head crop → 6DRepNet360 | uint8 BGR patch, variable size + box in frame | float32 `[1,3,224,224]`, RGB, ImageNet-normalized | resize, BGR→RGB, normalize, NCHW; the box carries forward |
| 6DRepNet360 → Gram-Schmidt | vector `[1,6]` | two 3D vectors | reshape `[6]` → `[3,2]` |
| Gram-Schmidt → angle to ray | R 3×3 in the dataset's own axes | unit vector in the optical frame (x right, y down, z forward) | fixed convention rotation, then face direction |
| camera_info → ray | `CameraInfo`: K, D, head pixel | undistorted pixel | `undistortPoints` with K and D |

### Adapters · B2 Iris and gaze

From the head crop to a gaze vector.

| Between | Out | In (expected) | Adapter |
|---|---|---|---|
| Head crop → MediaPipe | uint8 BGR patch + box in frame | `mp.Image` SRGB, uint8 H×W×3 | BGR→RGB, no normalization, pack into `mp.Image` |
| MediaPipe → eye gating | 478 landmarks, 0-1 relative to the crop | eye and iris points in frame pixels | × crop size, + box offset in frame |
| Gating → L2CS-Net | face bbox from landmarks + decision | float32 `[1,3,448,448]`, RGB, ImageNet-normalized | face crop from the original frame, resize, normalize |
| L2CS-Net → gaze-to-ray angle | 2 × 90 logits (yaw, pitch) | unit vector in the optical frame | softmax, Σ p·i × 4 − 180 → degrees → rad → 3D vector |
| Gaze + head → visible cone | two vectors, different sources | one vector + source + confidence | pick the source by priority, tag iris/head/body |

### Adapters · C Environment + D Aggregation

Streams at different frequencies, one state per ID.

| Between | Out | In (expected) | Adapter |
|---|---|---|---|
| Camera → SigLIP 2 | Image at ~15fps | float32 `[1,3,384,384]`, SigLIP normalization | throttle by timestamp to 1Hz, resize, mean/std 0.5 |
| SigLIP 2 → CBM head | embedding `[1,D]`; precomputed texts `[C,D]` | standardized score vector `[C]` | L2, cosine similarity, z-score with training mean/std |
| CBM head → topic | probabilities `[K]` + weights | `EnvContext`: place type, active concepts, reason | argmax, the concepts with the highest contribution |
| All cues → fusion | messages at 15Hz and 1Hz, different timestamps | per-ID per-tick: cue, value, confidence, age | approximate sync + "last value" cache; missing = unknown, not 0 |
