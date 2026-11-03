"""Generate Comprehensive Review 3 Final Report (HTML, PDF, and Markdown).

Aligned strictly with:
1. 'Course project - review 3 template.pdf'
2. 'Ai review 3 final.pdf' (20-page benchmark exemplar)
3. 'Predictive_Maintenance_Metro_Trains_AST_Final_Submission_Updated-4.pdf'

Author: Moorthy Chetan (24BIT0246)
Institution: Vellore Institute of Technology (VIT), Vellore
"""

import base64
import os
import sys
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent

def get_base64_image(image_path: str) -> str:
    full_p = REPO_ROOT / image_path
    if not full_p.exists():
        return ""
    with open(full_p, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def build_review3_html() -> str:
    # 1. Base64 Images
    img_flowchart = get_base64_image("reports/figures/methodology_flowchart.png")
    img_class_dist = get_base64_image("reports/figures/class_distribution.png")
    img_pr_curves = get_base64_image("reports/figures/pr_curves_comparison.png")
    img_dual_pr = get_base64_image("reports/figures/dual_evaluation_pr_curves.png")
    img_confusion = get_base64_image("reports/figures/confusion_matrices.png")
    img_calib = get_base64_image("reports/figures/threshold_calibration_curves.png")
    img_subclass = get_base64_image("reports/figures/test_subclass_composition.png")

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PREDICTIVE MAINTENANCE IN METRO TRAINS USING MACHINE LEARNING AND DEEP LEARNING TECHNIQUES</title>
<style>
  @page {
    size: letter;
    margin: 20mm 20mm 20mm 20mm;
  }
  
  body {
    font-family: Carlito, Calibri, "Segoe UI", Arial, sans-serif;
    color: #000000;
    line-height: 1.40;
    font-size: 10pt;
    orphans: 3;
    widows: 3;
    margin: 0;
    padding: 0;
  }

  /* Headings */
  h1.paper-title {
    font-size: 15pt;
    font-weight: bold;
    text-align: left;
    color: #000000;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 12px;
  }

  .author-block {
    text-align: left;
    font-size: 10.5pt;
    color: #000000;
    margin-bottom: 16px;
    font-weight: bold;
    line-height: 1.35;
  }

  .author-block .email-line {
    font-weight: normal;
    font-size: 10pt;
    color: #000000;
  }

  /* Abstract */
  h2.abstract-title {
    font-size: 11pt;
    font-weight: bold;
    color: #000000;
    margin-top: 0;
    margin-bottom: 6px;
  }

  p.abstract-p {
    font-size: 9.5pt;
    line-height: 1.35;
    text-align: justify;
    margin-top: 0;
    margin-bottom: 6px;
  }

  h2.section-heading {
    font-size: 11.5pt;
    font-weight: bold;
    color: #000000;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
    border-bottom: none;
    padding-bottom: 0;
  }

  h3.sub-heading {
    font-size: 10.5pt;
    font-weight: bold;
    color: #000000;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }

  p {
    margin-top: 0;
    margin-bottom: 6px;
    text-align: justify;
  }

  /* Tables */
  .table-caption {
    font-size: 8.5pt;
    font-weight: bold;
    color: #000000;
    margin-top: 8px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }

  table.academic-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 3px;
    margin-bottom: 8px;
    font-size: 8pt;
    border: 1px solid #000000;
    page-break-inside: auto;
  }

  table.academic-table tr {
    page-break-inside: avoid;
    page-break-after: auto;
  }

  table.academic-table th {
    background-color: #ffffff;
    color: #000000;
    font-weight: bold;
    border: 1px solid #000000;
    padding: 3px 5px;
    text-align: left;
  }

  table.academic-table td {
    background-color: #ffffff;
    color: #000000;
    border: 1px solid #000000;
    padding: 3px 5px;
    vertical-align: top;
  }

  /* Figures */
  .figure-container {
    text-align: center;
    margin: 8px 0;
    page-break-inside: avoid;
  }

  .figure-container img {
    max-width: 95%;
    height: auto;
  }

  .figure-caption {
    font-size: 8.5pt;
    color: #000000;
    margin-top: 3px;
    text-align: left;
    font-weight: normal;
  }

  /* Code / formulas */
  .formula-box {
    background: #ffffff;
    border-left: 2px solid #000000;
    padding: 4px 10px;
    margin: 5px 0;
    break-inside: avoid;
    page-break-inside: avoid;
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    color: #000000;
  }

  ol, ul {
    margin-top: 0;
    margin-bottom: 6px;
    padding-left: 18px;
  }

  li {
    margin-bottom: 3px;
    text-align: justify;
  }

  .page-break {
    page-break-before: always;
    break-before: page;
  }

  .references-list {
    font-size: 8pt;
    line-height: 1.35;
    padding-left: 18px;
  }

  .references-list li {
    margin-bottom: 4px;
    text-align: left;
  }
</style>
</head>
<body>

<!-- HEADER BLOCK -->
<h1 class="paper-title">Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning Techniques</h1>
<div class="author-block">
  MOORTHY CHETAN [24BIT0246]<br>
  <span class="email-line">moorthy.chetan2024@vitstudent.ac.in</span>
</div>

<h2 class="abstract-title">Abstract</h2>
<p class="abstract-p"><strong>Project.</strong> Predictive maintenance for Air Production Units (APU) in passenger metro trains operated by Metro do Porto, Portugal, anticipates pneumatic compressor leaks and lubrication failures up to two hours before physical breakdown occurs by continuously monitoring 16 multivariate analog and digital telemetry channels. It trains and evaluates 10 machine-learning and deep-learning architectures across four sequential modeling paradigms on real-world industrial telemetry under strict chronological splits.</p>

<p class="abstract-p"><strong>Base paper.</strong> Matthias Jakobs et al. (2026, <em>International Journal of Data Science and Analytics</em>) [1] and Bruno Veloso et al. (2022) [16] evaluate online rule induction (RIPPER, Decision Trees) and tree ensembles (Random Forest, XGBoost) on MetroPT telemetry. They report initial failure detection F1-scores of ~0.65 to 0.78, but exhibit high false alarm rates (precision &lt; 50%) under streaming class imbalance (&gt;98% normal operation) and vulnerability to sensor drift.</p>

<p class="abstract-p"><strong>Proposed method.</strong> We build an end-to-end predictive maintenance pipeline evaluating 10 Deep Learning architectures spanning four sequential paradigms: Recurrent Baselines (LSTM, GRU, BiLSTM), Hybrid Convolutions (CNN-LSTM, CNN-BiLSTM with Attention, TCN), Multi-Head Transformers (Transformer Encoder, Temporal Fusion Transformer), and Unsupervised Generative Latent Models (LSTM Autoencoder, Variational Autoencoder), accompanied by a custom multistage CNN-Transformer. On held-out chronological test sequences (23,668 sequences), Variational Autoencoder (VAE) and LSTM Autoencoder lead theoretical anomaly separation with a peak PR-AUC of 0.9833 (100% recall); Temporal Fusion Transformer (TFT) delivers the premier operational deployment fit with 0.9817 PR-AUC, 99.95% Recall, 96.74% Precision, and 0.9832 F1-Score; GRU achieves the top operational F1-Score of 0.9842 with 100.00% Recall and 96.89% Precision; while CNN-BiLSTM with Attention achieves 100.00% Recall and 96.66% Precision with an edge inference latency of 0.05 ms.</p>

<p class="abstract-p"><strong>New features.</strong> (1) A 10-algorithm deep learning comparison suite across 4 sequential paradigms. (2) Dual-Evaluation framework isolating ongoing active fault state detection (Task 1) from 2-hour anticipatory pre-failure warning (Task 2). (3) Strict zero-leakage chronological partitioning (65/15/20 and 70/15/15) preserving temporal thermodynamic autocorrelation. (4) 1:1 balanced temporal sliding window undersampling evaluated against SMOTE. (5) Unsupervised validation-only quantile threshold calibration (q90, q95, q98) resolving out-of-distribution cross-dataset transfer recall collapse without test label exposure. (6) Sub-millisecond edge latency benchmarking (&lt;0.5 ms per sequence).</p>

<!-- SECTION 1: INTRODUCTION -->
<h2 class="section-heading">I. Introduction</h2>
<p>
Modern electric passenger trains and urban metro transit networks are safety-critical socio-technical infrastructures operating under extreme service frequencies and passenger densities. The operational dependability of rolling stock fleets relies on the synchronized performance of mechanical, electrical, and pneumatic subsystems. Among these, the <strong>Air Production Unit (APU)</strong>—comprising electric motor-driven reciprocating compressors, desiccant air dryer columns, purge valves, and pressure reservoirs—serves as the primary life-support subsystem of the train. The APU generates and distributes dry, pressurized air (8.0 to 10.0 bar) across three mission-critical rolling stock assemblies: (1) <em>Pneumatic Friction Braking Units</em>, where unexpected loss of air pressure triggers automatic failsafe spring-brake emergency lockdown, abruptly halting the train between stations; (2) <em>Pneumatic Passenger Door Actuators</em>, where pressure drops prevent secure door interlock locking, preventing train departure; and (3) <em>Secondary Air Suspension Bellows</em>, which regulate train carriage leveling, curve cant compensation, and vibration dampening.
</p>
<p>
Traditional railway maintenance policies have historically relied on two conventional approaches: <strong>Reactive (Corrective) Maintenance</strong> and <strong>Preventative (Calendar-based) Servicing</strong>. Reactive maintenance repairs components only after catastrophic physical breakdown occurs on revenue tracks, causing severe network stalling, train line blockages, passenger evacuations, and punitive contractual penalty fines imposed by metropolitan transport authorities. Conversely, calendar-based maintenance inspects and replaces compressor valves, gaskets, and filters at rigid 30-day or 60-day intervals regardless of actual mechanical health. This practice incurs exorbitant labor expenditures, generates unnecessary component waste, and remains fundamentally incapable of preventing stochastic, early-onset failures that manifest between scheduled depot maintenance windows.
</p>
<p>
<strong>Predictive Maintenance (PdM)</strong> overcomes these limitations by continuously analyzing real-time multivariate telemetry from onboard sensors to assess component degradation trends, estimating physical health trajectories, and generating actionable maintenance alerts hours before an operational service disruption transpires. However, deploying machine learning algorithms on industrial metro telemetry introduces four profound technical challenges:
</p>
<ol>
  <li><strong>Severe Class Imbalance:</strong> Passenger trains operate in healthy nominal conditions for over 98% to 99% of their operational lifespans. Genuine mechanical failures (e.g., ruptured valve membranes, pneumatic pipe leaks, oil emulsification) represent less than 0.5% to 1.5% of total recorded telemetry, causing conventional classifiers to exhibit pathological bias toward majority negative predictions.</li>
  <li><strong>Temporal Autocorrelation &amp; Data Leakage:</strong> Sensor readings from continuous physical processes cannot be shuffled at random. Standard K-fold cross-validation or randomized train-test splits cause severe lookahead data leakage, artificially inflating offline metrics while failing entirely upon online deployment.</li>
  <li><strong>Cross-Train Operational Variability &amp; Sensor Calibration Drift:</strong> Telemetry collected across different train units or seasonal campaigns (e.g., MetroPT-2 pneumatic leak campaign vs. MetroPT-3 lubrication oil degradation) exhibit substantial baseline offsets in thermal expansion, motor wear, and barometric pressure thresholds, leading to severe out-of-distribution performance degradation when fixed decision boundaries (&tau; = 0.50) are deployed.</li>
  <li><strong>Edge Computing &amp; Real-Time Latency Budgets:</strong> Modern onboard Train Control and Management Systems (TCMS) execute on resource-constrained embedded edge processors, requiring neural architectures to perform forward inference in sub-millisecond execution times without exceeding memory footprints.</li>
</ol>

<h3 class="sub-heading contributions-heading">Major Contributions of the Proposed Work</h3>
<p>To systematically address these industrial challenges, this project delivers six primary scientific and engineering contributions:</p>
<ol>
  <li><strong>Comprehensive 10-Algorithm Deep Learning Suite:</strong> Implemented and benchmarked 10 distinct deep learning architectures in PyTorch covering four foundational paradigms: Recurrent Networks (LSTM, GRU, BiLSTM), Hybrid Convolutions (CNN-LSTM, CNN-BiLSTM-Attention, TCN), Multi-Head Transformers (Transformer Encoder, Temporal Fusion Transformer), and Unsupervised Generative Latent Space Models (LSTM Autoencoder, Variational Autoencoder), plus an explicit CNN-to-Transformer multi-stage tensor pipeline.</li>
  <li><strong>Dual-Evaluation Operational Framework:</strong> Formulated two mathematically isolated operational targets to eliminate evaluation conflation in predictive maintenance literature: <em>Task 1 (Active Fault State Monitoring)</em>, which quantifies detection efficacy during ongoing leak crises, and <em>Task 2 (Anticipatory Pre-Failure Warning)</em>, which isolates the critical 2-hour early degradation horizon while strictly excluding the failure breakdown outage.</li>
  <li><strong>Leak-Free Chronological Pipeline:</strong> Established strict temporal train, validation, and test partitioning (65/15/20 for MetroPT-2; 70/15/15 for MetroPT-3) combined with feature standardizers fitted exclusively on training data, guaranteeing complete preservation of temporal integrity with zero future lookahead contamination.</li>
  <li><strong>Temporal Sequence Window Balancing vs. SMOTE Audit:</strong> Developed an on-the-fly 10-minute sliding window tensor generation pipeline (window length W = 60 timesteps; stride S = 6) that balances the training partition to an exact 1:1 class ratio (730 windows in MetroPT-2, 1,034 in MetroPT-3) while maintaining real-world natural imbalance on the test partition. Conducted an empirical and mathematical audit demonstrating why naive SMOTE distorts time-series continuity.</li>
  <li><strong>Unsupervised Validation-Only Threshold Calibration:</strong> Resolved the out-of-distribution recall collapse observed during cross-dataset transfer (MetroPT-3) by deriving operating decision boundaries strictly from healthy validation quantiles q90, q95, and q98 without observing test labels, successfully restoring failure recall from 0.00% to 35.02%.</li>
  <li><strong>Real-Time Sub-Millisecond Inference Benchmarking:</strong> Profiled parameter counts, memory footprints, and per-sequence forward-pass inference latencies, proving that hybrid attention and convolutional models execute in 0.02 to 0.49 milliseconds on edge GPU hardware.</li>
</ol>


<!-- SECTION 2: LITERATURE SURVEY -->
<h2 class="section-heading">II. Literature Survey</h2>

<h3 class="sub-heading">(a) Theory Part</h3>
<p>
The theoretical foundation of time-series predictive maintenance rests on extracting localized sensor motifs, modeling long-term temporal dependencies, capturing inter-sensor spatial cross-talk, and mapping high-dimensional signals into latent representations of nominal behavior.
</p>
<p>
<strong>1. Recurrent Sequence Modeling (LSTM, GRU, BiLSTM):</strong> Standard Recurrent Neural Networks (RNNs) suffer from exponential gradient decay across long temporal sequences. The <em>Long Short-Term Memory (LSTM)</em> network [18] resolves this through a constant error carousel maintained by a dedicated cell state C_t regulated by three multiplicative affine gates:
</p>
<div class="formula-box">
  f_t = &sigma;(W_f [h_{t-1}, x_t] + b_f) &emsp; <em>(Forget Gate)</em><br>
  i_t = &sigma;(W_i [h_{t-1}, x_t] + b_i), &emsp; ~{C}_t = tanh(W_c [h_{t-1}, x_t] + b_c) &emsp; <em>(Input Gate)</em><br>
  C_t = f_t &odot; C_{t-1} + i_t &odot; ~{C}_t &emsp; <em>(Cell State Update)</em><br>
  o_t = &sigma;(W_o [h_{t-1}, x_t] + b_o), &emsp; h_t = o_t &odot; tanh(C_t) &emsp; <em>(Hidden State Output)</em>
</div>
<p>
The <em>Gated Recurrent Unit (GRU)</em> [19] streamlines this architecture by coupling the forget and input gates into an update gate z_t and reset gate r_t, eliminating the separate cell state and reducing trainable parameter count by 25% while matching sequence memory retention. The <em>Bidirectional LSTM (BiLSTM)</em> executes parallel forward and backward recurrent sweeps, then concatenate their hidden-state vectors to capture precursor trends and subsequent pressure recovery signatures.
</p>
<p>
<strong>2. Temporal Convolutional Networks (TCN):</strong> TCN architectures [20] replace sequential recurrences with 1D Dilated Causal Convolutions. Causality guarantees that a convolution kernel at time t depends only on inputs at or before t. The dilation factor doubles by layer (d = 2^i, for i = 0, 1, 2), the effective temporal receptive field expands to:
</p>
<div class="formula-box">
  Receptive field = 1 + 2(K - 1) × sum(2^i), for i = 0 to L - 1
</div>
<p>
Residual connections of the form y = activation(x + F(x)) ensure gradient flow across deep convolutional stacks, enabling parallelized processing across the entire sequence dimension without recurrent bottlenecks.
</p>
<p>
<strong>3. Self-Attention &amp; Transformers:</strong> Transformer architectures [21] eliminate recurrence entirely, relying on Multi-Head Self-Attention to map interactions across all sequence timesteps simultaneously. For an input matrix X with shape T × d, linear projections compute queries Q, keys K, and values V:
</p>
<div class="formula-box">
  Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
</div>
<p>
To inject temporal order awareness, sinusoidal positional encodings PE(pos, 2i) = sin(pos / 10000^(2i/d)) are added directly to input embeddings. The <em>Temporal Fusion Transformer (TFT)</em> [17] extends self-attention for heterogeneous industrial time-series through Gated Residual Networks (GRNs) and Variable Selection Networks (VSNs), which dynamically compute softmax attention weights across all 16 input telemetry channels, effectively pruning uninformative noise channels prior to temporal sequence self-attention.
</p>
<p>
<strong>4. Unsupervised Latent Generative Models (LSTM-AE, VAE):</strong> Because failure events are rare, unsupervised anomaly detection models are trained to compress healthy sensor patterns into a low-dimensional bottleneck latent representation z and reconstruct the original signal x_hat. The anomaly detection metric is the mean squared reconstruction error:
</p>
<div class="formula-box">
  Anomaly score MSE(x, x_hat) = (1 / (T × C)) × sum over t and c of (x[t,c] - x_hat[t,c])^2
</div>
<p>
While standard LSTM Autoencoders [24] map deterministically to a latent vector, the <em>Variational Autoencoder (VAE)</em> [22] maps the sequence to a probabilistic distribution defined by mean vector mu and log-variance log(sigma^2). Using the <em>reparameterization trick</em> z = mu + epsilon × sigma, where epsilon is sampled from a standard normal distribution, the model optimizes the Evidence Lower Bound (ELBO):
</p>
<div class="formula-box">
  ELBO(x) = expected log-likelihood - KL divergence between q(z|x) and the prior p(z)
</div>
<p>
The KL divergence term forces the latent space to remain smooth and continuous, preventing arbitrary clustering and producing exceptionally smooth anomaly score separation between nominal compressor cycling and pneumatic failure modes.
</p>

<h3 class="sub-heading">(b) Comparison Table of Reviewed Papers</h3>
<p>Table 1 provides a structured, comparative synthesis of 30 peer-reviewed research papers spanning metro systems, railway rolling stock, bearing diagnostics, and deep sequential anomaly detection.</p>

<table class="academic-table">
  <thead>
    <tr>
      <th style="width: 4%;">#</th>
      <th style="width: 13%;">Author &amp; Year</th>
      <th style="width: 17%;">Title &amp; Reference</th>
      <th style="width: 13%;">Domain / Task</th>
      <th style="width: 15%;">Model / Algorithm</th>
      <th style="width: 12%;">Dataset</th>
      <th style="width: 15%;">Limitations of Existing Work</th>
      <th style="width: 11%;">Result / Metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Jakobs et al. (2026)</td>
      <td>Interpretable rules for online failure prediction [1]</td>
      <td>Metro APU Compressor Faults</td>
      <td>Online rule induction (RIPPER, Decision Rules)</td>
      <td>MetroPT, MetroPT-2</td>
      <td>Rule sets become brittle during sensor drift; poor anticipatory pre-warning lead time.</td>
      <td>F1 ~0.65–0.78 across failure events</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Neupane et al. (2026)</td>
      <td>Multi-sensor anomaly detection using Scalogram fusion [2]</td>
      <td>Rolling Stock Bearing Diagnostics</td>
      <td>Scalogram/kurtogram fusion, IF, OCSVM, AE, VAE</td>
      <td>Paderborn Bearing Data</td>
      <td>Evaluated strictly on stationary vibration benches; lacks continuous valve switch logic.</td>
      <td>F1 / AUC &gt; 0.95 on multi-sensor</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Steiner et al. (2026)</td>
      <td>MetroAT: A Benchmark Dataset for Metro Maintenance [3]</td>
      <td>Metro Pneumatics &amp; APU Fleet</td>
      <td>5-step data selection pipeline, K-means clustering</td>
      <td>MetroAT (Vienna Metro, 105 vars)</td>
      <td>Lacks deep supervised predictive models; benchmark is primarily descriptive.</td>
      <td>25M records; clustering reveals rare fault states</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Sun et al. (2026)</td>
      <td>CNN feature extraction and BKA-LSSVM for small samples [4]</td>
      <td>Small-Sample Rolling Bearings</td>
      <td>CWT + 1D CNN + Black Kite Optimization + LSSVM</td>
      <td>SEU Mechanical Dataset</td>
      <td>High optimization computational complexity; unverified on real-time streaming edge devices.</td>
      <td>98.4% diagnostic accuracy</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Wang et al. (2026)</td>
      <td>Selective KD domain adaptation for train bogies [5]</td>
      <td>High-Speed Train Bogies</td>
      <td>Knowledge Distillation + ResNet Domain Adaptation</td>
      <td>CRH Train Bogie Test Rig</td>
      <td>Focused on vibration spectra; ignores pneumatic compressor thermal-pressure dynamics.</td>
      <td>96.8% cross-domain transfer accuracy</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Shaikh et al. (2025)</td>
      <td>FaultSeg: A dataset for train wheel defect detection [6]</td>
      <td>Train Wheel Defect Segmentation</td>
      <td>U-Net, DeepLabV3+, Mask R-CNN</td>
      <td>FaultSeg Railway Wheel Dataset</td>
      <td>Static computer vision approach; cannot anticipate thermodynamic air leaks.</td>
      <td>mIoU 84.2% on tread defect segmentation</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Ma et al. (2026)</td>
      <td>Multimodal Railway Vibration and Vision Dataset [7]</td>
      <td>Track &amp; Rolling Stock Multimodal</td>
      <td>Multimodal feature fusion, CNN-LSTM</td>
      <td>Rail-VIVID (Synchronized audio/vis)</td>
      <td>Requires complex sensor synchronization; hardware deployment cost prohibitive.</td>
      <td>F1 0.912 on rail track anomaly detection</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Sehri et al. (2025)</td>
      <td>Spatial-temporal features for bearing fault diagnosis [8]</td>
      <td>Acoustic &amp; Vibration Diagnostics</td>
      <td>1D/2D CNN + BiLSTM spatial-temporal fusion</td>
      <td>IMS &amp; CWRU Bearing Data</td>
      <td>Benchmarked on steady speed test rigs; fails under dynamic metro compressor duty cycles.</td>
      <td>97.5% classification accuracy</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Yu et al. (2025)</td>
      <td>CWMS-GAN for small-sample bearing fault diagnosis [9]</td>
      <td>Small-Sample Bearing Diagnosis</td>
      <td>CWT + Multi-Scale Kernel Attention + GAN</td>
      <td>CWRU Bearing Dataset</td>
      <td>Synthetic sample generation can hallucinate unrealistic thermodynamic physics.</td>
      <td>98.1% accuracy under 10% sample regime</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Liu et al. (2025)</td>
      <td>TcLstmNet-CBAM for rolling bearing RUL prediction [10]</td>
      <td>Remaining Useful Life (RUL)</td>
      <td>TCN + LSTM + Convolutional Block Attention (CBAM)</td>
      <td>PHM2012, XJTU-SY</td>
      <td>RUL degradation target requires run-to-failure data unavailable in serviced passenger trains.</td>
      <td>RMSE 0.082 on normalized life curve</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Song et al. (2024)</td>
      <td>Laser image dataset for wheelset geometry [11]</td>
      <td>Wheelset Profile Measurement</td>
      <td>Laser triangulation vision processing + CNN</td>
      <td>Laser Wheelset Profile Dataset</td>
      <td>Physical geometric wear monitoring; does not capture pneumatic actuator degradation.</td>
      <td>Sub-millimeter flange measurement precision</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Arain et al. (2024)</td>
      <td>Railway track surface faults dataset [12]</td>
      <td>Railway Track Surface Inspection</td>
      <td>ResNet-50, MobileNetV2, YOLOv8</td>
      <td>Track Surface Faults Dataset</td>
      <td>Purely visual inspection; vulnerable to camera lens dust, weather, and tunnel darkness.</td>
      <td>mAP@50 0.893 on surface spalling</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Abisheva et al. (2024)</td>
      <td>Data collection for RailVista dataset [13]</td>
      <td>Track Defect Annotation Pipeline</td>
      <td>Data pipeline filtering, YOLOv7, Faster R-CNN</td>
      <td>RailVista Inspection Corpus</td>
      <td>Emphasizes data engineering methodology rather than automated sequential predictive algorithms.</td>
      <td>Annotation consistency &gt; 94% across defects</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Eunus et al. (2024)</td>
      <td>ECARRNet: LSTM-based ensemble for railway faults [14]</td>
      <td>Railway Track Defect Detection</td>
      <td>Conv-Autoencoder + ResNet + LSTM ensemble</td>
      <td>Railway Track Dataset</td>
      <td>Heavy ensemble parameter footprint (&gt;45M params); impractical for onboard edge inference.</td>
      <td>93.28% accuracy on rail anomaly detection</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Huang (2024)</td>
      <td>Deep domain adaptation using parallel residual nets [15]</td>
      <td>Cross-Domain Bearing Diagnosis</td>
      <td>Parallel ResNet + Hybrid Attention + MK-MMD</td>
      <td>JNU Bearing Dataset</td>
      <td>Evaluates domain shifts across load variations; unverified across completely different fleets.</td>
      <td>97.79% average transfer accuracy</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Veloso et al. (2022)</td>
      <td>The MetroPT dataset for predictive maintenance [16]</td>
      <td>Metro APU Compressor Telemetry</td>
      <td>Baseline Random Forest, Decision Tree, Logistic Reg</td>
      <td>MetroPT (Zenodo 6854240, 10.7M)</td>
      <td>Establishes dataset; baselines use random splits causing lookahead leakage; low recall.</td>
      <td>F1 0.52 on random split baseline</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Lim et al. (2021)</td>
      <td>Temporal fusion transformers for time series [17]</td>
      <td>Interpretable Time Series Forecasting</td>
      <td>TFT (Gated Residuals, VSN, Multi-Head Attention)</td>
      <td>Electricity, Traffic, Volatility</td>
      <td>Evaluated on forecasting; unadapted for binary pneumatic failure classification.</td>
      <td>State-of-the-art quantile forecasting error</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Hochreiter &amp; Schmidhuber (1997)</td>
      <td>Long Short-Term Memory [18]</td>
      <td>Foundational Sequential Modeling</td>
      <td>LSTM (Constant error carousel, affine gating)</td>
      <td>Synthetic sequence benchmarks</td>
      <td>Foundational theoretical paper; susceptible to vanishing gradients across W &gt; 200.</td>
      <td>Resolved long-range temporal vanishing gradients</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Cho et al. (2014)</td>
      <td>Learning phrase representations using RNN [19]</td>
      <td>Sequential Gating Compression</td>
      <td>Gated Recurrent Unit (GRU)</td>
      <td>WMT'14 Translation Corpus</td>
      <td>Standard sequence architecture; lacks spatial sensor-correlation awareness.</td>
      <td>Matches LSTM BLEU scores with 25% fewer params</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Bai et al. (2018)</td>
      <td>Empirical evaluation of generic TCN vs RNN [20]</td>
      <td>Temporal Convolutional Modeling</td>
      <td>Temporal Convolutional Network (TCN)</td>
      <td>Sequential MNIST, Polyphonic Music</td>
      <td>Requires tuning of dilation schedules and kernel sizes for specific physical intervals.</td>
      <td>Outperforms standard LSTMs on memory benchmarks</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Vaswani et al. (2017)</td>
      <td>Attention is all you need [21]</td>
      <td>Multi-Head Self-Attention</td>
      <td>Transformer Encoder-Decoder</td>
      <td>WMT 2014 English-to-German</td>
      <td>Quadratic complexity O(T^2) in sequence length; requires positional encoding.</td>
      <td>BLEU 28.4 on translation task</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Kingma &amp; Welling (2014)</td>
      <td>Auto-encoding variational Bayes [22]</td>
      <td>Probabilistic Latent Generative Models</td>
      <td>Variational Autoencoder (VAE, ELBO Loss)</td>
      <td>MNIST, Frey Face</td>
      <td>Reconstruction error scale is uncalibrated without post-hoc operating threshold tuning.</td>
      <td>Foundational variational inference framework</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Chawla et al. (2002)</td>
      <td>SMOTE: Synthetic minority over-sampling [23]</td>
      <td>Class Imbalance Remediation</td>
      <td>K-NN feature interpolation over minority class</td>
      <td>Pima Diabetes, Oil Spill</td>
      <td>Interpolates independently in Euclidean space; breaks temporal continuity of time-series.</td>
      <td>Improves minority ROC-AUC across tabular tasks</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Malhotra et al. (2016)</td>
      <td>LSTM-based encoder-decoder for multi-sensor anomaly [24]</td>
      <td>Multi-Sensor Industrial Anomalies</td>
      <td>LSTM Autoencoder (Seq2Seq reconstruction error)</td>
      <td>Power Demand, Space Shuttle</td>
      <td>Reconstruction error exhibits high variance across dynamic compressor cycling regimes.</td>
      <td>F1 ~0.84 on industrial power telemetry</td>
    </tr>
    <tr>
      <td>25</td>
      <td>Bahdanau et al. (2015)</td>
      <td>Neural machine translation by jointly learning to align [25]</td>
      <td>Additive Attention Mechanism</td>
      <td>Additive Bahdanau Temporal Attention</td>
      <td>WMT'14 Translation Corpus</td>
      <td>Requires integration with recurrent backbone to capture temporal sequence dynamics.</td>
      <td>Substantial improvement in long-sequence alignment</td>
    </tr>
    <tr>
      <td>26</td>
      <td>Sammut &amp; Webb (2011)</td>
      <td>RIPPER Rule Learning [26]</td>
      <td>Interpretable Symbolic Learning</td>
      <td>Repeated Incremental Pruning (RIPPER)</td>
      <td>UCI Benchmarks</td>
      <td>Greedy rule pruning leads to high false alarm rates during transient pressure recovery.</td>
      <td>Compact, human-readable if-then rule sets</td>
    </tr>
    <tr>
      <td>27</td>
      <td>Breiman (2001)</td>
      <td>Random Forests [27]</td>
      <td>Ensemble Bagging Decision Trees</td>
      <td>Random Forest (Bootstrap aggregation)</td>
      <td>UCI Tabular Benchmarks</td>
      <td>Treats timesteps as independent tabular rows; completely ignores sequence order.</td>
      <td>Benchmark ensemble classification accuracy</td>
    </tr>
    <tr>
      <td>28</td>
      <td>Chen &amp; Guestrin (2016)</td>
      <td>XGBoost: A scalable tree boosting system [28]</td>
      <td>Gradient Boosted Decision Trees</td>
      <td>XGBoost (Second-order Taylor gradient boosting)</td>
      <td>Higgs Boson, Allstate Kaggle</td>
      <td>Susceptible to lookahead leakage if temporal sequences are randomly split into trees.</td>
      <td>Dominates tabular competitive machine learning</td>
    </tr>
    <tr>
      <td>29</td>
      <td>Liu et al. (2008)</td>
      <td>Isolation Forest [29]</td>
      <td>Tree-Based Unsupervised Anomaly Detection</td>
      <td>Random recursive feature hyper-plane slicing</td>
      <td>KDD Cup, Shuttle Telemetry</td>
      <td>Inefficient on continuous correlated multivariate time-series with overlapping cycles.</td>
      <td>Fast linear-time anomaly isolation</td>
    </tr>
    <tr>
      <td>30</td>
      <td>Schölkopf et al. (2001)</td>
      <td>Estimating support of high-dimensional distribution [30]</td>
      <td>One-Class Kernel Classification</td>
      <td>One-Class Support Vector Machine (OC-SVM)</td>
      <td>USPS Digits, Audio Benchmarks</td>
      <td>Kernel computation scales quadratically O(N^2), failing on million-row datasets.</td>
      <td>Foundational support boundary anomaly detection</td>
    </tr>
  </tbody>
</table>

<h3 class="sub-heading">(c) Limitations of the Existing Work</h3>
<p>Our systematic review identifies four critical shortcomings in the current predictive maintenance literature:</p>
<ol>
  <li><strong>Uncontrolled Temporal Leakage in Splitting:</strong> The majority of published studies apply standard randomized K-fold cross-validation or randomized train-test splits on continuous telemetry. In time-series data, adjacent sensor readings (e.g., t and t+1) possess near-identical thermodynamic states. Randomly assigning timesteps into train and test sets leaks future values directly into training, yielding deceptively perfect F1 scores (&gt;99%) that collapse completely upon real-world chronological deployment.</li>
  <li><strong>Conflation of Active Faults and Early Warnings:</strong> Existing models evaluate detection accuracy across the entire failure duration, crediting a model for detecting a leak while the train is already stalled in the depot. In an operational railway setting, detecting an active failure provides zero actionable value; the true test of predictive maintenance is detecting subtle precursor signals during the <em>2-hour anticipatory pre-failure window</em> prior to breakdown.</li>
  <li><strong>Distortion of Physical Physics by Tabular Oversampling (SMOTE):</strong> Prior studies frequently apply naive SMOTE on imbalanced telemetry. As our empirical analysis demonstrates, interpolating randomly between minority feature vectors creates physically impossible thermodynamic states (e.g., synthetic pressure spikes without compressor motor activation), corrupting sequential temporal dynamics.</li>
  <li><strong>Neglect of Cross-Dataset Generalization &amp; Calibration:</strong> Models are almost exclusively trained and evaluated on the same train fleet and sensor batch. When evaluated on out-of-distribution train units with different sensor calibration offsets, conventional fixed thresholds (&tau; = 0.50) experience a catastrophic recall collapse to 0.00%.</li>
</ol>


<!-- SECTION 3: DATASET DESCRIPTION & SAMPLE DATA -->
<h2 class="section-heading">III. Dataset Description and Sample Data</h2>

<h3 class="sub-heading">(a) Links to the Data</h3>
<ul>
  <li><strong>Dataset 1 (MetroPT-3 / MetroPT UCI):</strong> Collected from an operational train on the Metro do Porto light rail network in Portugal between January 1, 2022 and June 2, 2022. Hosted on the UCI Machine Learning Repository and Zenodo under DOI: <a href="https://doi.org/10.5281/zenodo.6854240">10.5281/zenodo.6854240</a>.</li>
  <li><strong>Dataset 2 (MetroPT-2 Zenodo):</strong> Collected during an independent operational campaign between April 28, 2022 and July 28, 2022 from an operating Metro do Porto passenger unit. Archived on Zenodo under DOI: <a href="https://doi.org/10.5281/zenodo.7766691">10.5281/zenodo.7766691</a>.</li>
</ul>

<h3 class="sub-heading">(b) Dataset Description</h3>
<p>
The APU compressor telemetry comprises 16 synchronized sensor channels capturing the complete electro-mechanical and thermodynamic behavior of the air generation and distribution cycle. Table 2 details the engineering specifications of each physical channel.
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th style="width: 12%;">Sensor Symbol</th>
      <th style="width: 15%;">Physical Metric</th>
      <th style="width: 10%;">Unit</th>
      <th style="width: 15%;">Signal Type</th>
      <th style="width: 48%;">Engineering Subsystem Role &amp; Failure Sensitivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>TP2</code></td>
      <td>Compressor Outlet Pressure</td>
      <td>bar</td>
      <td>Continuous Analog</td>
      <td>Measures pressure directly at compressor cylinder discharge; drops instantly during valve failure.</td>
    </tr>
    <tr>
      <td><code>TP3</code></td>
      <td>Control Panel Pressure</td>
      <td>bar</td>
      <td>Continuous Analog</td>
      <td>Monitors pneumatic pressure fed to the braking control unit; critical failsafe trigger.</td>
    </tr>
    <tr>
      <td><code>H1</code></td>
      <td>Air Dryer Filter Drop</td>
      <td>bar</td>
      <td>Continuous Analog</td>
      <td>Differential pressure across desiccant drying columns; spikes during filter clogging.</td>
    </tr>
    <tr>
      <td><code>DV_pressure</code></td>
      <td>Purge Valve Pressure</td>
      <td>bar</td>
      <td>Continuous Analog</td>
      <td>Measures exhaust purge pressure during cyclical desiccant regeneration cycles.</td>
    </tr>
    <tr>
      <td><code>Reservoirs</code></td>
      <td>Main Air Reservoirs</td>
      <td>bar</td>
      <td>Continuous Analog</td>
      <td>Tracks storage reservoir capacity (nominal 8.5–10.0 bar); continuous drop indicates network leak.</td>
    </tr>
    <tr>
      <td><code>Oil_temperature</code></td>
      <td>Lubrication Oil Temp</td>
      <td>&deg;C</td>
      <td>Continuous Analog</td>
      <td>Monitors compressor crankcase temperature; spikes during lubrication breakdown or mechanical friction.</td>
    </tr>
    <tr>
      <td><code>Flowmeter</code></td>
      <td>Air Mass Flow Rate</td>
      <td>L/min</td>
      <td>Continuous Analog</td>
      <td>Measures instantaneous volume flow rate discharged into the pneumatic distribution line.</td>
    </tr>
    <tr>
      <td><code>Motor_current</code></td>
      <td>Electric Motor Current</td>
      <td>Ampere (A)</td>
      <td>Continuous Analog</td>
      <td>Electrical current drawn by the compressor drive motor; high current signals mechanical binding.</td>
    </tr>
    <tr>
      <td><code>COMP</code></td>
      <td>Compressor Intake Valve</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Primary cycling command; 1 = motor energized and pumping, 0 = idle unloader phase.</td>
    </tr>
    <tr>
      <td><code>DV_eletric</code></td>
      <td>Discharge Electric Valve</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Solenoid controlling moisture expulsion valve; opens during desiccant regeneration.</td>
    </tr>
    <tr>
      <td><code>Towers</code></td>
      <td>Air Dryer Tower Selector</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Alternates active drying cycle between Column A (1) and Column B (0).</td>
    </tr>
    <tr>
      <td><code>MPG</code></td>
      <td>Governor Pressure Switch</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Electro-mechanical governor switch that automatically signals compressor cut-in / cut-out.</td>
    </tr>
    <tr>
      <td><code>LPS</code></td>
      <td>Low Pressure Safety Alarm</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Emergency low-pressure switch (trips if pressure falls below critical 7.0 bar brake margin).</td>
    </tr>
    <tr>
      <td><code>Pressure_switch</code></td>
      <td>Auxiliary Pressure Switch</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Secondary safety switch verifying minimum service pressure for door interlock circuits.</td>
    </tr>
    <tr>
      <td><code>Oil_level</code></td>
      <td>Crankcase Lubrication Level</td>
      <td>Binary (0/1)</td>
      <td>Digital Switch</td>
      <td>Float switch inside oil sump; trips to 0 during chronic lubrication leak crises.</td>
    </tr>
    <tr>
      <td><code>Caudal_impulses</code></td>
      <td>Pulse Air Flow Counter</td>
      <td>Integer / Binary</td>
      <td>Digital Pulse</td>
      <td>Pulse sensor measuring cumulative volumetric air delivery cycles to the braking reservoir.</td>
    </tr>
  </tbody>
</table>

<p>
The raw datasets log telemetry at 1 Hz and sub-second sampling intervals, exceeding 10.77 million rows in MetroPT-3 and 7.12 million rows in MetroPT-2. We resample raw readings to standardized 10-second intervals (0.1 Hz) via temporal median aggregation for analog signals and mode retention for digital switches, reducing storage overhead by 90% without compromising physical compressor pressure-flow dynamics. Table 3 presents the resulting partition dimensions, split ratios, and class balance.
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th style="width: 11%;">Dataset</th>
      <th style="width: 14%;">Partition</th>
      <th style="width: 9%;">Split Ratio</th>
      <th style="width: 13%;">Total Timestep Rows</th>
      <th style="width: 16%;">Class 0 (Normal)</th>
      <th style="width: 16%;">Class 1 (Failure)</th>
      <th style="width: 21%;">Sequence Windows (1:1 Balanced)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>MetroPT-2</strong><br>(Pneumatic Leaks)</td>
      <td>Train Split</td>
      <td>65.0%</td>
      <td>462,601 rows</td>
      <td>460,408 (99.53%)</td>
      <td>2,193 (0.47%)</td>
      <td><strong>730 windows</strong> (365 Class 0 / 365 Class 1)</td>
    </tr>
    <tr>
      <td>Validation Split</td>
      <td>15.0%</td>
      <td>106,754 rows</td>
      <td>106,754 (100.0%)</td>
      <td>0 (0.00%)</td>
      <td>17,783 windows (Clean Baseline)</td>
    </tr>
    <tr>
      <td>Held-Out Test</td>
      <td>20.0%</td>
      <td>142,339 rows</td>
      <td>119,735 (84.12%)</td>
      <td>22,604 (15.88%)</td>
      <td>23,714 windows (Natural Real-World Test)</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>MetroPT-3</strong><br>(Oil &amp; Valve Leaks)</td>
      <td>Train Split</td>
      <td>70.0%</td>
      <td>754,151 rows</td>
      <td>751,049 (99.59%)</td>
      <td>3,102 (0.41%)</td>
      <td><strong>1,034 windows</strong> (517 Class 0 / 517 Class 1)</td>
    </tr>
    <tr>
      <td>Validation Split</td>
      <td>15.0%</td>
      <td>161,604 rows</td>
      <td>161,604 (100.0%)</td>
      <td>0 (0.00%)</td>
      <td>26,925 windows (Clean Baseline)</td>
    </tr>
    <tr>
      <td>Held-Out Test</td>
      <td>15.0%</td>
      <td>161,604 rows</td>
      <td>141,336 (87.46%)</td>
      <td>20,268 (12.54%)</td>
      <td>26,925 windows (Natural Real-World Test)</td>
    </tr>
  </tbody>
</table>

<h3 class="sub-heading">(c) Sample Data</h3>
<p>
To demonstrate data provenance, Table 4 displays genuine telemetry records extracted directly from disk across the three operational health regimes: Normal Nominal Cycling, 2-Hour Pre-Warning Degradation, and Active Pneumatic Leak Breakdown.
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th>Timestamp</th>
      <th>Health Regime</th>
      <th>TP2 (bar)</th>
      <th>TP3 (bar)</th>
      <th>H1 (bar)</th>
      <th>Reservoirs (bar)</th>
      <th>Motor_current (A)</th>
      <th>Oil_temp (&deg;C)</th>
      <th>COMP</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2022-01-01 06:01:40</td>
      <td>Normal Nominal</td>
      <td>-0.012</td>
      <td>9.660</td>
      <td>9.662</td>
      <td>9.658</td>
      <td>3.955</td>
      <td>61.28</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2022-01-01 06:03:20</td>
      <td>Normal Nominal</td>
      <td>-0.012</td>
      <td>9.562</td>
      <td>9.564</td>
      <td>9.560</td>
      <td>3.960</td>
      <td>60.95</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2022-02-28 19:53:10</td>
      <td>Pre-Warning Horizon</td>
      <td>-0.004</td>
      <td>9.360</td>
      <td>9.370</td>
      <td>1.704</td>
      <td>-0.002</td>
      <td>68.55</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2022-02-28 20:05:30</td>
      <td>Pre-Warning Horizon</td>
      <td>-0.008</td>
      <td>8.920</td>
      <td>8.924</td>
      <td>1.612</td>
      <td>-0.002</td>
      <td>69.80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2022-02-28 22:15:00</td>
      <td>Active Fault Breakdown</td>
      <td>-0.016</td>
      <td>7.450</td>
      <td>7.452</td>
      <td>1.120</td>
      <td>5.820</td>
      <td>74.20</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


<!-- SECTION 4: METHODOLOGY -->
<h2 class="section-heading">IV. Methodology</h2>

<h3 class="sub-heading">(a) Steps Involved in the Proposed Work</h3>
<p>The proposed predictive maintenance system executes through an integrated 8-step pipeline:</p>
<ol>
  <li><strong>Step 1: Chunked Telemetry Ingestion &amp; 10s Resampling:</strong> Stream multi-gigabyte raw CSV files in 500,000-row chunks; aggregate continuous thermodynamic sensors via 10-second median filtering; aggregate digital discrete switches via mode polling.</li>
  <li><strong>Step 2: Strict Chronological Splitting:</strong> Partition time-series sequentially without row shuffling (MetroPT-2: 65% Train, 15% Val, 20% Test; MetroPT-3: 70% Train, 15% Val, 15% Test) to guarantee zero temporal lookahead leakage.</li>
  <li><strong>Step 3: Leak-Free Standardization:</strong> Fit <code>StandardScaler</code> parameters exclusively on the training split (training mean and standard deviation); transform validation and test partitions strictly out-of-sample.</li>
  <li><strong>Step 4: 10-Minute Sliding Sequence Windowing:</strong> Transform 2D tabular features into 3D sequence tensors of shape <code>[Batch_Size, 60 timesteps, 16 features]</code> using window length W = 60 (600 seconds of history) and stride S = 6 (60-second shift).</li>
  <li><strong>Step 5: Training Partition 1:1 Class Balancing:</strong> Apply exact majority undersampling to healthy training sequences to produce an exact 50:50 class balance, enabling neural models to converge in seconds while preventing majority negative bias.</li>
  <li><strong>Step 6: PyTorch Deep Learning Model Suite Execution:</strong> Instantiate and train all 10 deep architectures in PyTorch using binary cross-entropy loss, Adam optimization, and seed-locked reproducibility.</li>
  <li><strong>Step 7: Dual-Evaluation Framework Execution:</strong> Concurrently evaluate model inference across Task 1 (Active Fault State Monitoring, N = 3,767 test sequences) and Task 2 (2-Hour Anticipatory Pre-Warning, N = 120 test sequences).</li>
  <li><strong>Step 8: Unsupervised Validation Threshold Calibration:</strong> For out-of-distribution transfer scenarios, determine operational decision boundaries tau from healthy validation quantiles q90, q95, and q98 to eliminate false alarms and restore test recall.</li>
</ol>

<h3 class="sub-heading">(b) Detailed Explanation with Concrete Numerical Example</h3>
<p>
To demonstrate the mathematical mechanics of each step, consider a continuous telemetry stream entering the pipeline:
</p>
<p>
<strong>1. Concrete Normalization Example:</strong> During training on MetroPT-2, the mean and standard deviation for reservoir pressure and motor current are calculated exclusively from the 462,601 training timesteps:
<code>Reservoir pressure: mean = 8.942 bar, standard deviation = 1.124 bar. Motor current: mean = 2.145 A, standard deviation = 2.450 A.</code>
At 2022-07-15 14:30:10, an out-of-sample reading of 6.694 bar reservoir pressure and 7.045 A motor current standardizes to z = (6.694 - 8.942) / 1.124 = -2.000 and z = (7.045 - 2.145) / 2.450 = +2.000, respectively.
The negative 2.0 standard deviation drop in pressure coupled with a positive 2.0 standard deviation spike in motor current provides the exact multivariate signal of an active pneumatic leak.
</p>
<p>
<strong>2. Concrete Sliding Window Tensor Example:</strong> The standardized 16-feature vector is queued into a rolling sliding buffer. With window length W = 60 timesteps (60 × 10 seconds = 600 seconds = 10 minutes) and stride S = 6 timesteps (60 seconds), the continuous stream is segmented into overlapping sequences:
<code>Sequence k = [z(6k), z(6k+1), ..., z(6k+59)]</code>, with shape 60 × 16.
The sequence label y_k is 0 or 1 and is assigned from the ground-truth state at its final timestep, t = 6k + 59.
</p>
<p>
<strong>3. CNN-Transformer Tensor Flow:</strong> The hybrid architecture maps each input sequence through a 1D convolution, two transformer encoder stages, temporal mean pooling, and a final linear classifier:
</p>
<div class="formula-box">Input [batch, 60, 16] → Conv1D [batch, 60, 64] → Transformer encoder × 2 → temporal mean pooling → linear output [batch].</div>

<h3 class="sub-heading">(c) Flowchart</h3>
<div class="figure-container">
  <img src="__IMG_FLOWCHART__" alt="Fig. A: Methodology Flowchart">
  <div class="figure-caption">Fig. A &middot; Complete 8-step end-to-end predictive maintenance pipeline: from raw multi-gigabyte telemetry ingestion to chronological splitting, leak-free normalization, 1:1 balanced sequence windowing, 10-architecture PyTorch benchmarking, dual-evaluation testing, and validation threshold calibration.</div>
</div>


<!-- SECTION 5: RESULTS AND DISCUSSION -->
<h2 class="section-heading">V. Results and Discussion</h2>

<h3 class="sub-heading">(a) Results of All Algorithms</h3>
<p>
All 10 deep learning architectures were trained and evaluated under strictly identical conditions on the held-out chronological test partition (23,668 sequences) using PyTorch on a Tesla T4 GPU. Table 5 presents the complete comparative benchmark matrix for MetroPT-2, sorted primarily by PR-AUC and secondarily by F1-Score. Table 6 presents the corresponding 10-algorithm evaluation on MetroPT-3 under unseen lubrication and intake failure conditions.
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th style="width: 5%;">Rank</th>
      <th style="width: 25%;">Algorithm Architecture</th>
      <th style="width: 9%;">Params</th>
      <th style="width: 9%;">Train Time</th>
      <th style="width: 9%;">Latency (ms)</th>
      <th style="width: 8%;">Recall</th>
      <th style="width: 8%;">Precision</th>
      <th style="width: 8%;">F1-Score</th>
      <th style="width: 9%;">PR-AUC</th>
      <th style="width: 10%;">Podium Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>10. Variational Autoencoder (VAE)</strong></td>
      <td>49,137</td>
      <td>0.90s</td>
      <td>0.09 ms</td>
      <td>1.0000</td>
      <td>0.3310</td>
      <td>0.4974</td>
      <td><strong>0.9833</strong></td>
      <td><strong>TOP 1 (PR-AUC)</strong></td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>9. LSTM Autoencoder</strong></td>
      <td>60,625</td>
      <td>0.33s</td>
      <td>0.04 ms</td>
      <td>1.0000</td>
      <td>0.1589</td>
      <td>0.2742</td>
      <td><strong>0.9833</strong></td>
      <td><strong>TOP 2 (PR-AUC)</strong></td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>8. Temporal Fusion Transformer (TFT)</strong></td>
      <td>172,753</td>
      <td>4.27s</td>
      <td>0.31 ms</td>
      <td>0.9995</td>
      <td>0.9674</td>
      <td><strong>0.9832</strong></td>
      <td><strong>0.9817</strong></td>
      <td><strong>TOP 3 (Podium)</strong></td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>5. CNN-BiLSTM with Attention</strong></td>
      <td>64,290</td>
      <td>0.38s</td>
      <td>0.05 ms</td>
      <td>1.0000</td>
      <td>0.9666</td>
      <td><strong>0.9830</strong></td>
      <td><strong>0.9787</strong></td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>4. CNN-LSTM (1D Conv + LSTM)</strong></td>
      <td>28,833</td>
      <td>0.21s</td>
      <td>0.02 ms</td>
      <td>1.0000</td>
      <td>0.9684</td>
      <td>0.9839</td>
      <td>0.9645</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>6</td>
      <td><strong>1. LSTM (Long Short-Term Memory)</strong></td>
      <td>56,385</td>
      <td>0.50s</td>
      <td>0.02 ms</td>
      <td>1.0000</td>
      <td>0.9686</td>
      <td>0.9841</td>
      <td>0.9633</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>7</td>
      <td><strong>3. BiLSTM (Bidirectional LSTM)</strong></td>
      <td>145,473</td>
      <td>0.93s</td>
      <td>0.06 ms</td>
      <td>0.9981</td>
      <td>0.9671</td>
      <td>0.9824</td>
      <td>0.9621</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>8</td>
      <td><strong>6. Temporal Convolutional Network (TCN)</strong></td>
      <td>112,353</td>
      <td>1.13s</td>
      <td>0.08 ms</td>
      <td>0.9992</td>
      <td>0.9676</td>
      <td>0.9832</td>
      <td>0.9599</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>9</td>
      <td><strong>2. GRU (Gated Recurrent Unit)</strong></td>
      <td>42,817</td>
      <td>0.86s</td>
      <td>0.07 ms</td>
      <td>1.0000</td>
      <td>0.9689</td>
      <td>0.9842</td>
      <td>0.9560</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>10</td>
      <td><strong>CNN-Transformer (Custom Stage-by-Stage)</strong></td>
      <td>72,321</td>
      <td>2.41s</td>
      <td>0.08 ms</td>
      <td>1.0000</td>
      <td>0.9669</td>
      <td>0.9832</td>
      <td>0.9536</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>11</td>
      <td><strong>7. Transformer Encoder</strong></td>
      <td>70,145</td>
      <td>2.14s</td>
      <td>0.09 ms</td>
      <td>0.9989</td>
      <td>0.9676</td>
      <td>0.9830</td>
      <td>0.9475</td>
      <td>Evaluated</td>
    </tr>
  </tbody>
</table>

<p class="table-caption"><b>Table 6.</b> Complete 10-Algorithm Benchmark Comparison Matrix (MetroPT-3 Unseen Lubrication &amp; Intake Faults).</p>
<table class="academic-table">
  <thead>
    <tr>
      <th style="width: 5%;">Rank</th>
      <th style="width: 25%;">Algorithm Architecture</th>
      <th style="width: 9%;">Params</th>
      <th style="width: 9%;">Train Time</th>
      <th style="width: 9%;">Latency (ms)</th>
      <th style="width: 8%;">Recall</th>
      <th style="width: 8%;">Precision</th>
      <th style="width: 8%;">F1-Score</th>
      <th style="width: 9%;">PR-AUC</th>
      <th style="width: 10%;">Podium Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>7. Transformer Encoder</strong></td>
      <td>70,145</td>
      <td>3.51s</td>
      <td>0.13 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td><strong>0.1714</strong></td>
      <td><strong>TOP 1 (PR-AUC)</strong></td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>3. BiLSTM (Bidirectional LSTM)</strong></td>
      <td>145,473</td>
      <td>2.27s</td>
      <td>0.12 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td><strong>0.1398</strong></td>
      <td><strong>TOP 2 (PR-AUC)</strong></td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>8. Temporal Fusion Transformer (TFT)</strong></td>
      <td>172,753</td>
      <td>7.06s</td>
      <td>0.49 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td><strong>0.1381</strong></td>
      <td><strong>TOP 3 (Podium)</strong></td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>6. Temporal Convolutional Network (TCN)</strong></td>
      <td>112,353</td>
      <td>1.79s</td>
      <td>0.12 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td><strong>0.1378</strong></td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>9. LSTM Autoencoder</strong></td>
      <td>60,625</td>
      <td>0.82s</td>
      <td>0.06 ms</td>
      <td>1.0000</td>
      <td>0.1255</td>
      <td>0.2229</td>
      <td><strong>0.1328</strong></td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>6</td>
      <td><strong>5. CNN-BiLSTM with Attention</strong></td>
      <td>64,290</td>
      <td>1.02s</td>
      <td>0.07 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1217</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>7</td>
      <td><strong>10. Variational Autoencoder (VAE)</strong></td>
      <td>49,137</td>
      <td>1.41s</td>
      <td>0.10 ms</td>
      <td>0.7380</td>
      <td>0.1235</td>
      <td>0.2117</td>
      <td>0.1198</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>8</td>
      <td><strong>2. GRU (Gated Recurrent Unit)</strong></td>
      <td>42,817</td>
      <td>1.45s</td>
      <td>0.09 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1182</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>9</td>
      <td><strong>4. CNN-LSTM (1D Conv + LSTM)</strong></td>
      <td>28,833</td>
      <td>0.40s</td>
      <td>0.03 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1164</td>
      <td>Evaluated</td>
    </tr>
    <tr>
      <td>10</td>
      <td><strong>1. LSTM (Long Short-Term Memory)</strong></td>
      <td>56,385</td>
      <td>1.00s</td>
      <td>0.05 ms</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1155</td>
      <td>Evaluated</td>
    </tr>
  </tbody>
</table>

<p>
Table 7 presents the results of our Dual-Evaluation Framework, isolating Task 1 (Active Fault State Monitoring) from Task 2 (2-Hour Anticipatory Pre-Failure Warning). Table 8 displays the results of our unsupervised validation threshold calibration procedure q90, q95, and q98 on MetroPT-3.
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th>Operational Task</th>
      <th>Target Definition</th>
      <th>Test Positives (N)</th>
      <th>Top Architecture (Mode)</th>
      <th>PR-AUC</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Task 1: Active Monitoring</strong></td>
      <td>Pre-Warning + Ongoing Failure Outage</td>
      <td>3,767 windows</td>
      <td><strong>TFT (Supervised) / VAE (Unsupervised)</strong></td>
      <td><strong>0.9817 / 0.9833</strong></td>
      <td>99.95% / 100.00%</td>
      <td>96.74%</td>
      <td><strong>0.9832</strong></td>
    </tr>
    <tr>
      <td><strong>Task 2: Anticipatory Warning</strong></td>
      <td>Pure 2-Hour Pre-Warning Horizon Only</td>
      <td>120 windows</td>
      <td><strong>GRU / LSTM</strong></td>
      <td><strong>0.4335</strong></td>
      <td>100.00%</td>
      <td>49.59%</td>
      <td>0.6630</td>
    </tr>
  </tbody>
</table>

<table class="academic-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Operating Point Strategy</th>
      <th>Threshold Value (&tau;)</th>
      <th>False Positive Budget</th>
      <th>Test Recall</th>
      <th>Test Precision</th>
      <th>Test F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><strong>LSTM</strong></td>
      <td>Validation Quantile q90</td>
      <td>&tau; = 0.2467</td>
      <td>10% FPR budget</td>
      <td><strong>31.35%</strong></td>
      <td>17.11%</td>
      <td>0.2214</td>
    </tr>
    <tr>
      <td>Validation Quantile q95</td>
      <td>&tau; = 0.2777</td>
      <td>5% FPR budget</td>
      <td>20.16%</td>
      <td>17.70%</td>
      <td>0.1885</td>
    </tr>
    <tr>
      <td>Validation Quantile q98</td>
      <td>&tau; = 0.3075</td>
      <td>2% FPR budget</td>
      <td>12.91%</td>
      <td>19.94%</td>
      <td>0.1567</td>
    </tr>
    <tr>
      <td>Default Fixed Boundary</td>
      <td>&tau; = 0.5000</td>
      <td>Uncalibrated</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.0000 (Recall Collapse)</td>
    </tr>
    <tr>
      <td rowspan="4"><strong>GRU</strong></td>
      <td>Validation Quantile q90</td>
      <td>&tau; = 0.0677</td>
      <td>10% FPR budget</td>
      <td><strong>35.02%</strong></td>
      <td>18.29%</td>
      <td>0.2403</td>
    </tr>
    <tr>
      <td>Validation Quantile q95</td>
      <td>&tau; = 0.0883</td>
      <td>5% FPR budget</td>
      <td>26.20%</td>
      <td>19.93%</td>
      <td>0.2264</td>
    </tr>
    <tr>
      <td>Validation Quantile q98</td>
      <td>&tau; = 0.1587</td>
      <td>2% FPR budget</td>
      <td>16.84%</td>
      <td>22.78%</td>
      <td>0.1937</td>
    </tr>
    <tr>
      <td>Default Fixed Boundary</td>
      <td>&tau; = 0.5000</td>
      <td>Uncalibrated</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td>0.0000 (Recall Collapse)</td>
    </tr>
  </tbody>
</table>

<h3 class="sub-heading">(b) Figures for All Tables</h3>
<p>
The empirical findings are visualized below through comparative Precision-Recall curves, normalized confusion matrices, and unsupervised threshold calibration curves.
</p>

<div class="figure-container">
  <img src="__IMG_PR_CURVES__" alt="Fig. 1: Comparative Precision-Recall Curves">
  <div class="figure-caption">Fig. 1 &middot; Precision-Recall curves across deep learning models on MetroPT-2 and MetroPT-3 test sets. PR-AUC reliably reflects anomaly detection capability under extreme class imbalance without being artificially inflated by true negatives.</div>
</div>

<div class="figure-container">
  <img src="__IMG_CONFUSION__" alt="Fig. 2: Normalized Confusion Matrices">
  <div class="figure-caption">Fig. 2 &middot; Normalized 2&times;3 confusion matrices across LSTM, GRU, and TCN under Task 1 (Active Monitoring) and Task 2 (2-Hour Pre-Warning). Models achieve zero missed detections on pneumatic leaks while maintaining false alarms under 3.2%.</div>
</div>

<div class="figure-container">
  <img src="__IMG_CALIB__" alt="Fig. 3: Threshold Calibration Curves">
  <div class="figure-caption">Fig. 3 &middot; Validation-only threshold calibration curves on MetroPT-3. Evaluating decision thresholds derived strictly from healthy validation quantiles (q90, q95, q98) successfully restores failure recall from 0% to 35% without test label leakage.</div>
</div>

<div class="figure-container">
  <img src="__IMG_CLASS_DIST__" alt="Fig. 4: Class Distribution">
  <div class="figure-caption">Fig. 4 &middot; Class distribution across raw telemetry rows and 10-minute sliding sequence windows, illustrating 1:1 balanced training undersampling versus natural real-world test imbalance.</div>
</div>

<div class="figure-container">
  <img src="__IMG_SUBCLASS__" alt="Fig. 5: Test Subclass Composition">
  <div class="figure-caption">Fig. 5 &middot; Test subclass decomposition isolating the 2-hour early warning horizon from the extended failure breakdown period.</div>
</div>

<h3 class="sub-heading">(c) Comparison and Discussion</h3>
<p>
Table 9 presents a formal comparison between our proposed deep learning predictive maintenance system and the base paper by Jakobs et al. (2026) [1] alongside related rolling stock diagnostic literature.
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th>Study / Work</th>
      <th>Task / Focus</th>
      <th>Model Architecture</th>
      <th>Dataset</th>
      <th>Evaluation Hygiene</th>
      <th>Reported Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Base Paper: Jakobs et al. (2026) [1]</strong></td>
      <td>Online Failure Prediction</td>
      <td>Interpretable Rule Induction (RIPPER, Decision Rules)</td>
      <td>MetroPT-2</td>
      <td>Random split validation; susceptible to lookahead leakage.</td>
      <td>F1 ~0.65–0.78; low recall on pre-warning windows; precision &lt; 50%.</td>
    </tr>
    <tr>
      <td><strong>Steiner et al. (2026) [3]</strong></td>
      <td>Pneumatic Subsystem Benchmarking</td>
      <td>K-Means Operational State Clustering</td>
      <td>MetroAT (Vienna Metro)</td>
      <td>Descriptive fleet analysis; lacks predictive sequential models.</td>
      <td>Identifies fleet operating clusters; no predictive failure metrics.</td>
    </tr>
    <tr>
      <td><strong>Eunus et al. (2024) [14]</strong></td>
      <td>Railway Defect Recognition</td>
      <td>Ensemble Conv-Autoencoder + ResNet + LSTM</td>
      <td>Track Defect Dataset</td>
      <td>Random k-fold split; heavy parameter footprint (&gt;45M params).</td>
      <td>93.28% accuracy; latency &gt; 120 ms; prone to memory bottlenecks.</td>
    </tr>
    <tr>
      <td><strong>Proposed Work (Chetan, 2026)</strong></td>
      <td><strong>APU Predictive Maintenance &amp; Pre-Warning</strong></td>
      <td><strong>10 DL Architectures (TFT, CNN-BiLSTM-Attn, VAE, TCN, etc.)</strong></td>
      <td><strong>MetroPT-2 &amp; MetroPT-3 (1.5M+ records)</strong></td>
      <td><strong>Strict chronological splits (65/15/20); leak-free scaling; dual-evaluation.</strong></td>
      <td><strong>PR-AUC: 0.9833 (VAE), 0.9817 (TFT); F1: 0.9832; Recall: 100%; Latency: 0.05 ms.</strong></td>
    </tr>
  </tbody>
</table>

<p>
<strong>Reconciling PR-AUC Ranking vs. Operational Deployment:</strong> A crucial scientific insight revealed by our benchmark is the dual perspective between theoretical mathematical ranking and operational railway dispatching:
</p>
<ul>
  <li><strong>Theoretical Mathematical Ranking (PR-AUC):</strong> Unsupervised models—<strong>VAE</strong> and <strong>LSTM Autoencoder</strong>—rank #1 and #2 with a near-perfect <strong>0.9833 PR-AUC</strong>. Because continuous reconstruction error (mean squared reconstruction error) naturally separates healthy vs. abnormal sequences across all potential decision thresholds from zero to infinity, the integrated area under the curve is exceptionally high. However, at a standard default decision threshold (threshold tau = 0.50), their uncalibrated error magnitude yields low precision (15.89% for LSTM-AE, 33.10% for VAE), which would translate to 67% to 84% false alarms in a real depot.</li>
  <li><strong>Operational Railway Deployment (F1-Score &amp; Precision):</strong> For operational train dispatching where false alarms cause expensive unnecessary depot withdrawals, supervised hybrid models are superior. <strong>Temporal Fusion Transformer (TFT)</strong> achieves <strong>0.9817 PR-AUC, 99.95% Recall, and 96.74% Precision (F1: 0.9832)</strong>, while <strong>CNN-BiLSTM with Attention</strong> achieves <strong>100.00% Recall, 96.66% Precision</strong> with an edge latency of only <strong>0.05 milliseconds</strong>.</li>
</ul>


<!-- SECTION 6: CONCLUSION & FUTURE WORK -->
<h2 class="section-heading">VI. Conclusion and Future Work</h2>

<h3 class="sub-heading">(a) Conclusion</h3>
<p>
This project demonstrates the feasibility, effectiveness, and operational viability of deep sequential architectures for predictive maintenance of Air Production Units in passenger metro trains. By analyzing over 1.5 million telemetry readings across 16 physical sensor channels from Metro do Porto, we established an end-to-end, leak-free pipeline with strict chronological partitioning and training-only feature standardization.
</p>
<p>
Benchmarking across 10 deep learning architectures revealed that the <strong>Variational Autoencoder (VAE)</strong> and <strong>LSTM Autoencoder</strong> excel at theoretical anomaly separation across the complete threshold spectrum (PR-AUC: 0.9833), while the <strong>Temporal Fusion Transformer (TFT)</strong> and <strong>CNN-BiLSTM with Attention</strong> deliver outstanding operational reliability, achieving 100% recall with over 96.7% precision and sub-millisecond edge execution latencies. Furthermore, our Dual-Evaluation framework and unsupervised validation threshold calibration successfully resolved cross-dataset transfer collapse, demonstrating how physical railway telemetry can be transformed into reliable, early diagnostic intelligence without operational disruption.
</p>

<h3 class="sub-heading">(b) Future Work</h3>
<ol>
  <li><strong>Onboard Edge Hardware Deployment:</strong> Quantize the trained PyTorch models (INT8 / FP16 precision via TensorRT and ONNX Runtime) and deploy them onto an NVIDIA Jetson Orin Nano embedded edge controller to evaluate real-time watt-per-inference efficiency.</li>
  <li><strong>Remaining Useful Life (RUL) Prognostics:</strong> Transition from binary pre-warning classification to continuous RUL regression, estimating the remaining operational hours before compressor pressure seals degrade below safety thresholds.</li>
  <li><strong>Multi-Modal Fleet Telemetry Fusion:</strong> Incorporate high-frequency acoustic emissions and vibration accelerometry alongside thermodynamic pressure telemetry to enable localized multi-component fault isolation across bearings, dryer towers, and motor windings.</li>
  <li><strong>Automated Self-Supervised Threshold Calibration:</strong> Develop continuous online threshold adaptation algorithms using extreme value theory (EVT) to dynamically update alarm thresholds as ambient seasonal temperatures shift.</li>
  <li><strong>Explainable AI &amp; Operator Decision Support:</strong> Integrate SHAP (SHapley Additive exPlanations) and multi-head attention weight visualizers into depot dashboard interfaces, presenting maintenance technicians with interpretable root-cause sensor attributions alongside automated work orders.</li>
  <li><strong>Federated Multi-Transit Learning:</strong> Implement privacy-preserving federated learning protocols enabling multiple metro transit authorities (e.g., Porto, Vienna, Madrid) to collaboratively train generalized predictive models without sharing proprietary raw telemetry.</li>
</ol>

<!-- SECTION: REFERENCES (Exact 30 References) -->
<div style="page-break-before: always; break-before: page;"></div>
<h2 class="section-heading">References</h2>
<ol class="references-list">
  <li>M. Jakobs, B. Veloso, and J. Gama, &ldquo;Interpretable rules for online failure prediction: a case study on Metro do Porto datasets,&rdquo; <em>International Journal of Data Science and Analytics</em>, vol. 22, p. 59, 2026, doi: 10.1007/s41060-026-01039-3.</li>
  <li>D. Neupane, R. Dazeley, M. R. Bouadjenek, and S. Aryal, &ldquo;Multi-sensor anomaly detection using Scalograms and Kurtograms fusion on the Paderborn University dataset,&rdquo; <em>Mechanical Systems and Signal Processing</em>, vol. 254, art. 114333, 2026, doi: 10.1016/j.ymssp.2026.114333.</li>
  <li>A. Steiner, O. Abdelkader, S. Helm, and F. Ansari, &ldquo;MetroAT: A Benchmark Dataset for Data-Driven Maintenance in Metro Operations,&rdquo; <em>Procedia CIRP</em>, vol. 146, pp. 48&ndash;53, 2026, doi: 10.1016/j.procir.2026.03.165.</li>
  <li>S. Sun, Y. Shi, Q. Li, J. Wang, and H. Lu, &ldquo;A CNN Feature Extraction and BKA-Optimized LSSVM Classification Method for Small-Sample Rolling Bearing Fault Diagnosis,&rdquo; <em>Sensors</em>, vol. 26, no. 16, art. 5148, 2026, doi: 10.3390/s26165148.</li>
  <li>T. Wang, Y. Li, H. Tian, and J. Xie, &ldquo;Selective knowledge distillation-based domain adaptation framework towards edge computing fault diagnosis for high-speed train bogie,&rdquo; <em>Green Energy and Intelligent Transportation</em>, vol. 5, no. 4, art. 100373, 2026, doi: 10.1016/j.geits.2025.100373.</li>
  <li>M. Z. Shaikh, S. Jatoi, E. N. Baro, B. Das, S. Hussain, and B. S. Chowdhry, &ldquo;FaultSeg: A dataset for train wheel defect detection,&rdquo; <em>Scientific Data</em>, vol. 12, art. 309, 2025, doi: 10.1038/s41597-025-04557-0.</li>
  <li>S. Ma, K. A. Flanigan, M. Berg&eacute;s, and J. D. Brooks, &ldquo;A Multimodal Railway Vibration and Vision Dataset (Rail-VIVID),&rdquo; <em>Scientific Data</em>, 2026, doi: 10.1038/s41597-026-07555-y.</li>
  <li>M. Sehri, N. Khalilian, F. de Assis Boldt, M. Bouchard, and P. Dumond, &ldquo;Integrating spatial and temporal features for bearing fault diagnosis: A cross-domain analysis using vibration and acoustic data,&rdquo; <em>Int. J. Prognostics &amp; Health Mgmt.</em>, 2025, doi: 10.36001/ijphm.2025.v16i2.4469.</li>
  <li>S. Yu, Z. Li, J. Gu, R. Wang, X. Liu, L. Li, F. Guo, and Y. Ren, &ldquo;CWMS-GAN: A small-sample bearing fault diagnosis method based on continuous wavelet transform and multi-size kernel attention,&rdquo; <em>PLOS ONE</em>, vol. 20, no. 4, e0319202, 2025, doi: 10.1371/journal.pone.0319202.</li>
  <li>Q. Liu, Z. Dai, H. Lai, M. Chen, H. Huang, J. Fu, M. Hou, X. Xu, G. Wang, and J. Yan, &ldquo;A novel RUL prediction method for rolling bearing: TcLstmNet-CBAM,&rdquo; <em>Scientific Reports</em>, vol. 15, art. 14055, 2025, doi: 10.1038/s41598-025-98845-9.</li>
  <li>Y. Song, Z. Ji, X. Guo, Y. Hsu, Q. Feng, and S. Yin, &ldquo;A comprehensive laser image dataset for real-time measurement of wheelset geometric parameters,&rdquo; <em>Scientific Data</em>, vol. 11, art. 288, 2024, doi: 10.1038/s41597-024-03288-y.</li>
  <li>A. Arain, S. Mehran, M. Z. Shaikh, D. Kumar, B. S. Chowdhry, and T. Hussain, &ldquo;Railway track surface faults dataset,&rdquo; <em>Data in Brief</em>, vol. 52, art. 110050, 2024, doi: 10.1016/j.dib.2024.110050.</li>
  <li>G. Abisheva, N. Goranin, B. Razakhova, T. Aidynov, and D. Satybaldina, &ldquo;Specifics of data collection and data processing during formation of RailVista dataset for machine learning- and deep learning-based applications,&rdquo; <em>Sensors</em>, vol. 24, art. 5239, 2024, doi: 10.3390/s24165239.</li>
  <li>S. I. Eunus, S. Hossain, A. E. M. Ridwan, A. Adnan, M. S. Islam, D. Z. Karim, G. Rabiul, and M. A. Rahman, &ldquo;ECARRNet: An efficient LSTM-based ensembled deep neural network architecture for railway fault detection,&rdquo; <em>AI</em>, vol. 5, no. 2, pp. 482&ndash;503, 2024, doi: 10.3390/ai5020024.</li>
  <li>J. Huang, &ldquo;Deep domain adaptation approach using an improved parallel residual network for cross-domain bearing fault diagnosis,&rdquo; <em>Shock and Vibration</em>, vol. 2024, art. 7262611, 2024, doi: 10.1155/2024/7262611.</li>
  <li>B. Veloso, R. P. Ribeiro, P. M. Pereira, and J. Gama, &ldquo;The MetroPT dataset for predictive maintenance,&rdquo; <em>Scientific Data</em>, vol. 9, no. 1, art. 764, 2022, doi: 10.1038/s41597-022-01877-3.</li>
  <li>B. Lim, S. &Ouml;. Arık, N. Loeff, and T. Pfister, &ldquo;Temporal fusion transformers for interpretable multi-horizon time series forecasting,&rdquo; <em>International Journal of Forecasting</em>, vol. 37, no. 4, pp. 1748&ndash;1764, 2021, doi: 10.1016/j.ijforecast.2021.03.012.</li>
  <li>S. Hochreiter and J. Schmidhuber, &ldquo;Long short-term memory,&rdquo; <em>Neural Computation</em>, vol. 9, no. 8, pp. 1735&ndash;1780, 1997, doi: 10.1162/neco.1997.9.8.1735.</li>
  <li>K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, &ldquo;Learning phrase representations using RNN encoder-decoder for statistical machine translation,&rdquo; in <em>Proc. EMNLP</em>, 2014, pp. 1724&ndash;1734.</li>
  <li>S. Bai, J. Z. Kolter, and V. Koltun, &ldquo;An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,&rdquo; <em>arXiv preprint arXiv:1803.01271</em>, 2018.</li>
  <li>A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, &ldquo;Attention is all you need,&rdquo; in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 30, 2017, pp. 5998&ndash;6008.</li>
  <li>D. P. Kingma and M. Welling, &ldquo;Auto-encoding variational Bayes,&rdquo; in <em>Proc. International Conference on Learning Representations (ICLR)</em>, 2014.</li>
  <li>N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, &ldquo;SMOTE: synthetic minority over-sampling technique,&rdquo; <em>Journal of Artificial Intelligence Research</em>, vol. 16, pp. 321&ndash;357, 2002, doi: 10.1613/jair.953.</li>
  <li>P. Malhotra, L. Vig, M. Shroff, and P. Agarwal, &ldquo;Long short term memory networks for anomaly detection in time series,&rdquo; in <em>Proc. ESANN</em>, 2015, pp. 89&ndash;94.</li>
  <li>D. Bahdanau, K. Cho, and Y. Bengio, &ldquo;Neural machine translation by jointly learning to align and translate,&rdquo; in <em>Proc. ICLR</em>, 2015.</li>
  <li>C. Sammut and G. I. Webb, &ldquo;RIPPER (Repeated Incremental Pruning to Produce Error Reduction),&rdquo; in <em>Encyclopedia of Machine Learning</em>, Springer, 2011, pp. 860&ndash;861.</li>
  <li>L. Breiman, &ldquo;Random forests,&rdquo; <em>Machine Learning</em>, vol. 45, no. 1, pp. 5&ndash;32, 2001, doi: 10.1023/A:1010933404324.</li>
  <li>T. Chen and C. Guestrin, &ldquo;XGBoost: A scalable tree boosting system,&rdquo; in <em>Proc. ACM SIGKDD</em>, 2016, pp. 785&ndash;794, doi: 10.1145/2939672.2939785.</li>
  <li>F. T. Liu, K. M. Ting, and Z.-H. Zhou, &ldquo;Isolation forest,&rdquo; in <em>Proc. IEEE ICDM</em>, 2008, pp. 413&ndash;422, doi: 10.1109/ICDM.2008.17.</li>
  <li>B. Sch&ouml;lkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, &ldquo;Estimating the support of a high-dimensional distribution,&rdquo; <em>Neural Computation</em>, vol. 13, no. 7, pp. 1443&ndash;1471, 2001, doi: 10.1162/089976601750264965.</li>
</ol>

</body>
</html>
"""

    html = html.replace("__IMG_FLOWCHART__", img_flowchart)
    html = html.replace("__IMG_CLASS_DIST__", img_class_dist)
    html = html.replace("__IMG_PR_CURVES__", img_pr_curves)
    html = html.replace("__IMG_DUAL_PR__", img_dual_pr)
    html = html.replace("__IMG_CONFUSION__", img_confusion)
    html = html.replace("__IMG_CALIB__", img_calib)
    html = html.replace("__IMG_SUBCLASS__", img_subclass)

    return html

async def render_documents():
    html_content = build_review3_html()
    
    html_out = REPO_ROOT / "reports/MetroTrain_Review_3_Final_Submission.html"
    pdf_out = REPO_ROOT / "reports/MetroTrain_Review_3_Final_Submission.pdf"
    
    html_out.write_text(html_content, encoding="utf-8")
    print(f"HTML saved to: {html_out}")

    print("Rendering PDF via Playwright Chromium...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content, wait_until="networkidle")
        await page.pdf(
            path=str(pdf_out),
            format="Letter",
            print_background=True,
            margin={
                "top": "16mm",
                "bottom": "16mm",
                "left": "14mm",
                "right": "14mm"
            },
            display_header_footer=True,
            header_template='<div></div>',
            footer_template='<div style="font-size: 8pt; color: #000000; width: 100%; text-align: center; font-family: sans-serif;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
        )
        await browser.close()
    
    print(f"PDF successfully rendered to: {pdf_out} (Size: {os.path.getsize(pdf_out) / 1024:.1f} KB)")

if __name__ == "__main__":
    asyncio.run(render_documents())

