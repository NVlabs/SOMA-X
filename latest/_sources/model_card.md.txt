# Model Overview

## Description:
SOMA (Unifying Parametric Human Body Models) is a unified framework that decouples identity representation from pose parameterization by mapping any supported parametric body model to a single canonical mesh topology and skeleton, enabling a shared Linear Blend Skinning (LBS) pipeline across all backends. SOMA supports six identity backends (SOMA-shape, SMPL, SMPL-X, MHR, ANNY, and GarmentMeasurements), unified under a canonical mesh topology and SOMA skeleton. 

This model is ready for commercial use. 

## License/Terms of Use:
SOMA is released under the [Apache 2.0 License](../LICENSE).

## Deployment Geography:
Global

## Use Case:
SOMA is intended for use by computer vision researchers, graphics and animation engineers, machine learning engineers, and robotics researchers and companies. Specific use cases include:
- **Pose estimation and human reconstruction** — unified pose interface enables seamless identity substitution across backends without retraining.
- **Motion generation and animation** — apply motion capture sequences to any supported identity model using the same axis-angle pose parameterization.
- **Avatar synthesis and digital humans** — freely mix identity sources with SOMA's pose representation.
- **Simulation and robotics** — lightweight analytical forward pass enables real-time simulation pipelines with diverse body shapes.

## Expected Release Date:
GitHub: 03/16/2026 <br>

## Reference(s):
- SOMA: Unifying Parametric Human Body Models.
- SMPL: A Skinned Multi-Person Linear Model — Loper et al., 2015
- SMPL-X: Expressive Body Capture: 3D Hands, Face, and Body from a Single Image — Pavlakos et al., 2019
- MHR: Momentum Human Rig, Meta
- ANNY: Anthropometric body model spanning full human lifespan

## Model Architecture:
**Architecture Type:** Analytical / Parametric; optional shallow Multilayer Perceptron (MLP) for pose-dependent surface correctives <br>

**Network Architecture:** The core pipeline uses no learned neural network components. It is composed of three closed-form analytical modules:
1. **Barycentric Mesh Transfer** — sparse barycentric correspondence matrix pre-computed per backend; runtime topology transfer is a single sparse matrix-vector product in O(V_h) time.
2. **RBF Skeleton Fitting** — Radial Basis Function regression with Kabsch rotation alignment yields the 77-joint identity-adapted skeleton transforms in a single linear solve per identity.
3. **Linear Blend Skinning (LBS)** — standard LBS with joint-orient (T-pose-relative) parameterization; GPU-accelerated via NVIDIA Warp custom kernels with `torch.export`-compatible interface.

Optional: shallow pose-dependent corrective MLP (2 hidden layers, ReLU activations) for surface artifact reduction. <br>

**This model was developed independently by NVIDIA.** <br>

**Number of model parameters:**
- Core analytical pipeline: 0 learned parameters (closed-form)
- Principal Component Analysis (PCA) for SOMA-shape model: 128 principal components x ~18,000 mid-LOD vertices x 3 = ~3.1 x 10^6 coefficients (pre-fitted, not gradient-trained)
- Optional pose corrective Multilayer Perceptron (MLP) layers: ~1 × 10^8 parameters (if enabled)

## Computational Load
**Throughput:** > 7,033 posed meshes/second on NVIDIA A100 80GB (batch size 128, GPU Warp path) <br>
**Latency:** 2.1 ms per mesh (batch = 1, GPU); 12.1 ms (batch = 1, CPU 32-core) <br>
**Skeleton fitting:** < 1.68 ms (batch = 1) <br>
**Training compute:** N/A — core pipeline requires no gradient-based training; PCA shape space fitted offline from body scan data.

## Input(s):
**Input Type(s):** Numerical tensors (floating-point) <br>

**Input Format(s):**
- Identity coefficients: floating-point tensor, shape `(B, K)` where `K = 128` for SOMA-shape backend or backend-specific dimensionality for SMPL/SMPL-X/MHR/ANNY/Garment
- Pose parameters: axis-angle vectors `(B, 77, 3)` or rotation matrices `(B, 77, 3, 3)` covering 77 articulated joints (excludes root dummy joint)
- Optional root translation: `(B, 3)` in meters

**Input Parameters:** One-Dimensional (1D) coefficient vectors; Three-Dimensional (3D) pose tensors <br>

**Other Properties Related to Input:**
- Identity coefficients should lie within the shape space of the respective backend (no hard clipping, but extreme out-of-distribution values may produce artifact geometry).
- Pose parameters follow standard axis-angle convention; no clamping is applied.
- All inputs are standard float32 PyTorch tensors. No pre-processing beyond normalization within each backend's identity model is required.

## Output(s):
**Output Type(s):** Numerical tensors (3D geometry) <br>

**Output Format(s):** PyTorch float32 tensors <br>

**Output Parameters:** Three-Dimensional (3D) <br>

- Posed mesh vertices: `(B, N_h, 3)` where `N_h` depends on `lod` (`mid`: 18,056; `low`: 4,505; `xlo`: 612) -- world-space vertex positions in **meters**
- Joint positions: `(B, 77, 3)` — world-space 3D joint positions in **meters**
- Rest-shape vertices: `(B, N_h, 3)` in meters, with the same LOD-dependent `N_h` (intermediate output, available on request)

**Other Properties Related to Output:** All outputs are in meters. Body vertex count `N_h` is selected by `SOMALayer(lod=...)`; `low_lod=True` is the legacy alias for `lod="low"`. Joint count is fixed at 77.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (GPU cores) and software frameworks (CUDA libraries, NVIDIA Warp), the model achieves real-time throughput exceeding 7,000 meshes per second at batch size 128 on an A100 GPU.

## Software Integration:
**Runtime Engine(s):**
* NVIDIA Warp (GPU-accelerated LBS kernel, `torch.export`-compatible) <br>
* PyTorch (CPU and GPU fallback) <br>
* N/A — No dependency on TAO, Riva, NeMo, or other NVIDIA SDK runtimes

**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere (A100, A30, A40, A10, RTX 3000-series) — tested on A100 80GB <br>
* NVIDIA Hopper (H100) — forward compatible via Warp/CUDA <br>
* NVIDIA Ada Lovelace (RTX 4000-series, L40) <br>
* NVIDIA Turing (T4, RTX 2000-series) <br>
* NVIDIA Volta (V100) <br>
* Any NVIDIA GPU with CUDA support — model is lightweight and runs on any NVIDIA GPU <br>
* CPU only (PyTorch fallback, no CUDA required)

**Preferred/Supported Operating System(s):**
* Linux <br>
* Windows (via PyTorch CPU/GPU path) <br>

## Model Version(s):
- **SOMA v1.0** — initial public release; includes full-body layer (`SOMALayer`, 77 joints, mid/low/xlo LODs) and all six identity backends.

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

**SOMA-shape Identity Model (Shape PCA):**
- **SizeUSA** — commercially licensed 3D body scan dataset; largest anthropometric survey of the U.S. population, covering diverse body shapes across age, sex, and BMI groups. Used to compute the 128-component PCA shape space for the SOMA-native identity backend.
- **TripleGangers** — commercially licensed 3D body scan dataset purchased from TripleGangers, containing body scans of 303 individuals. Contributes additional shape diversity to the SOMA-shape PCA.
- **GarmentMeasurement PCA model** — body shape data distilled from the GarmentMeasurement parametric model to augment the shape space with garment-relevant proportions.

**Shallow MLP for Pose Correctives:** 

- **Bones RigPlay Dataset**: 80,000 (pose,mesh) pairs samples from Bones RigPlay motion capture dataset owned by NVIDIA. 

**Data Modality** 
- Other: 3D meshes and 3D motion data 

**Non-Audio, Image, Text Training Data Size**: 20,000 3D meshes and 80,000 (pose,mesh) pairs.  <br>

**Data Collection Method:** Automatic/Sensors (structured-light 3D body scanners) for SizeUSA and TripleGangers; Synthetic for GarmentMeasurement distillation; motion capture for Bones RigPlay. <br>
**Labeling Method:** Automatic/Sensors (body landmarks auto-detected from scans) <br>

**Dataset License(s):** SizeUSA — commercially licensed (purchased by NVIDIA); TripleGangers — commercially licensed (purchased by NVIDIA); GarmentMeasurement — Generated using the source code; Bones RigPlay - commercially licensed (purchased by NVIDIA)

## Testing Dataset:
Not applicable 

## Evaluation Dataset:
Not applicable

Data Collection Method: Synthetic <br>
Labeling Method: Automatic <br>
**Dataset License(s):** N/A

# Inference:
**Acceleration Engine:** NVIDIA Warp (custom LBS kernel, `torch.export`-compatible); PyTorch (fallback) <br>
**Test Hardware:**
* NVIDIA A100 80GB (primary GPU benchmark hardware)
* 32-core AMD EPYC 7763 (CPU benchmark hardware)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

SOMA produces 3D human body meshes in a purely geometric and anonymous form; it does not process images, video, or any personally identifiable data at inference time. Input shape coefficients do not correspond to real individuals unless explicitly constructed to do so. Developers integrating SOMA into applications that may reconstruct or represent real individuals should ensure they have obtained appropriate consent and comply with applicable privacy regulations.

For more detailed information on ethical considerations for this model, please see the Model Card++ subcards: [BIAS.md](BIAS.md), [EXPLAINABILITY.md](EXPLAINABILITY.md), [SAFETY_and_SECURITY.md](SAFETY_and_SECURITY.md), and [PRIVACY.md](PRIVACY.md).

Please report model quality, risk, or security vulnerabilities [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
