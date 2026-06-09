# Prompt for IEEE Paper Writing

You are an expert academic writer specialising in machine learning, federated learning, and weather forecasting. Your task is to write a complete, submission-ready IEEE conference paper in LaTeX (for https://www.escience-conference.org/2026/call-for-papers). This paper is a direct scientific evolution of a previous workshop paper by the same authors. Read every detail below carefully before writing anything.

---

## 1. Previous Paper (to evolve, not repeat)

The previous paper was published at BigHPC2025 (ITADATA2025, CEUR-WS Vol-4124, paper51):
**"Federated Learning for Distributed Weather Forecasting: A Practical Approach on Real Multidimensional Georeferenced Data"**
Authors: Attilio Di Vicino, Giuseppe Fiorillo, Luigi Galluccio, Raffaele Montella
University of Naples "Parthenope", Department of Science and Technology

Key facts about the previous paper:
- Used 4 weather stations, 20,199 data points, ~1-hour sampling interval
- Target variable: temperature
- Features: station_id, temperature, humidity, wind_speed, wind_sin, wind_cos (4 input features)
- Models: Transformer and Crossformer, both with d_model=64, nhead=4, num_layers=2, dropout=0.1
- input_window=20, output_window=1
- Infrastructure: Signal K + MQTT, Docker, Hadoop on Microsoft Azure
- FL framework: Flower with FedProx (20 rounds, 25 local epochs per round)
- Normalisation: zero mean, unit variance (standard scaling)
- Best result: Crossformer MAE=0.144, MSE=0.232
- Transformer: MAE=0.194 (reported as 0.313 vs De Vita baseline), MSE=0.265
- Compared against De Vita et al. [IEEE e-Science 2024] baseline: Crossformer MSE=1.324 MAE=0.827

---

## 2. What This New Paper Contributes (the evolution)

This new paper extends the previous work in the following concrete ways:

### 2.1 Significantly expanded dataset
- **8 weather stations** (doubled from 4), all operated by University of Naples "Parthenope"
- Station IDs: ws1 through ws8
- **~46,214 hourly rows** after preprocessing (vs 20,199 in the previous paper)
- **Date range: January 2025 – June 2026** (~17 months, vs ~months in previous paper)
- Raw data volume: ~4.37 million sub-minute readings before hourly resampling
- Station row counts after resampling: ws1=6136, ws2=4889, ws3=6245, ws4=5273, ws5=6203, ws6=5653, ws7=4450, ws8=7365

### 2.2 Improved data preprocessing pipeline
The new pipeline introduces:
1. **Hourly resampling**: raw sub-minute data (~10 second intervals) aggregated to 1-hour means. Wind direction encoded as sin/cos BEFORE resampling (averaging vector components, not angles — mathematically correct).
2. **Rigorous feature selection based on data quality analysis**:
   - Dropped: BarTrend (100% zeros, zero variance), SolarRad (78.9% missing), UV (85.2% missing), RainRate (99% zeros — original target candidate rejected), Barometer retained (8.3% missing — imputable)
   - Kept: Barometer, HumOut, RainDay, WindSpeed, WindSpeed10Min, wind_sin, wind_cos, station_id
3. **Physical validity clamping**: sensor readings outside physical ranges set to NaN before imputation
4. **Per-station imputation**: forward-fill (limit 3h) then linear interpolation (limit 6h), then column median for residuals
5. **Global temporal sort**: dataset sorted by datetime globally (not per-station) ensuring correct train/val/test temporal splits
6. **MinMax normalisation** to [0,1]

### 2.3 Target variable change
- Previous paper: temperature (TempOut) — same target, but now with richer features
- New paper: **TempOut** remains the target, but RainRate was evaluated and rejected (99% zeros)
- Input features: station_id, Barometer, HumOut, RainDay, WindSpeed, WindSpeed10Min, wind_sin, wind_cos → **8 input features** (vs 4 in previous paper)

### 2.4 Correct temporal evaluation methodology
- Previous paper used a simple row-index split (which caused station-leakage between splits)
- New paper uses **per-station temporal split** applied before window creation, then ConcatDataset — ensures each sliding window contains only consecutive hours from one station
- Split ratios: centralised 80/10/10, federated per-client 70/15/15
- All 8 stations present in every split

### 2.5 Warm-start federated learning
- New paper introduces **warm-start FL**: the global model is initialised from the centralised trained checkpoint before FL rounds begin
- This is a key methodological contribution: clients start from a good global model (centralised Skill~0.76) rather than random weights, making FL rounds serve their intended purpose of local adaptation rather than recovering from random init
- proximal_mu reduced to 0.01 (from 0.1 in previous paper) to allow meaningful local adaptation
- local_epochs=5 per FL round (vs 25 in previous paper, but 5 is per-round here with warm start)
- rounds=20

### 2.6 Infrastructure update
- Previous: Hadoop on Microsoft Azure
- New: **NVIDIA RTX 4080 16GB GPU** (local HPC node at University of Naples "Parthenope")
- PyTorch AMP (Automatic Mixed Precision) for training efficiency
- Flower framework retained with FedProx

### 2.7 Model configuration changes
- input_window=20, output_window=2 (was output_window=1)
- Both models: d_model=64, nhead=4, num_layers=2, dropout=0.1 (same as before, kept small due to dataset size)
- Centralised training uses ReduceLROnPlateau scheduler (factor=0.5, patience=10) and early stopping (patience=20)
- batch_size=128, lr=0.001

---

## 3. Complete Experimental Results

### 3.1 Centralised Training Results (fixed 80/10/10 temporal split)

**Transformer (centralised):**
- Best val loss: 0.02627
- MAE = 0.1433
- RMSE = 0.1838
- Skill Score = 0.6956

**Crossformer (centralised):**
- Best val loss: 0.01971
- MAE = 0.1227
- RMSE = 0.1639
- Skill Score = 0.7578

Note: Crossformer outperforms Transformer in centralised setting (consistent with previous paper).

### 3.2 Federated Learning Results (warm-start FedProx, 20 rounds)

**Crossformer — Federated (per-round aggregated metrics):**

| Round | MAE      | RMSE     | Skill Score |
|-------|----------|----------|-------------|
| 1     | 0.180755 | 0.241892 | 0.320616    |
| 5     | 0.220002 | 0.283456 | 0.084781    |
| 10    | 0.227595 | 0.290579 | 0.083156    |
| 15    | 0.289475 | 0.345819 | -0.270917   |
| 20    | 0.274943 | 0.334208 | -0.194335   |

Best federated Crossformer: Round 1 (MAE=0.1808, RMSE=0.2419, Skill=0.3206)

**Transformer — Federated (per-round aggregated metrics):**

| Round | MAE      | RMSE     | Skill Score |
|-------|----------|----------|-------------|
| 1     | 0.326290 | 0.373128 | -0.656473   |
| 5     | 0.390055 | 0.419123 | -1.046707   |
| 10    | 0.416201 | 0.444915 | -1.178950   |
| 15    | 0.426188 | 0.445728 | -1.220789   |
| 20    | 0.384462 | 0.404933 | -0.967290   |

Best federated Transformer: Round 1 (MAE=0.3263, RMSE=0.3731, Skill=-0.6565)

### 3.3 Metric definitions
- **MAE**: Mean Absolute Error (on MinMax-scaled [0,1] data; multiply by 56 for °C approximation)
- **RMSE**: Root Mean Squared Error (same scale)
- **Skill Score**: 1 − MSE_model / MSE_baseline where baseline = always predicting zero. Positive = better than naive baseline; negative = worse.
- MSE is not reported directly; RMSE² gives MSE if needed.

### 3.4 Key observations for the Discussion section
1. **Centralised Crossformer > centralised Transformer** (consistent with previous paper finding)
2. **Warm-start FL works for Crossformer at round 1** (Skill=0.32) but degrades over rounds — suggesting client drift with FedProx µ=0.01 and heterogeneous station data
3. **Warm-start FL does not work for Transformer** in this configuration — Skill is negative throughout, indicating the Transformer global model is more sensitive to federated aggregation than Crossformer
4. **Performance gap centralised vs federated**: Crossformer centralised Skill=0.758 → federated best Skill=0.321. This gap is the main finding to discuss.
5. **FL degrades over rounds** (not improves) for both models — this is a known challenge with non-IID federated data across stations with different local climates and sensor characteristics. It is worth discussing in the context of data heterogeneity.
6. The warm-start approach is a methodological contribution regardless of whether FL improves over centralised — it establishes a meaningful starting point and shows how much performance is sacrificed for privacy preservation.

---

## 4. Authors and Affiliations

- Giuseppe Fiorillo (giuseppe.fiorillo001@studenti.uniparthenope.it) — ORCID: 0009-0005-4769-0920
- Attilio Di Vicino (attilio.divicino001@studenti.uniparthenope.it) — ORCID: 0009-0009-8159-9118
- Luigi Galluccio (luigi.galluccio001@studenti.uniparthenope.it) — ORCID: 0009-0002-8441-3381
- Diana Di Luccio – ORCID: 0000-0002-0810-2250 (diana.diluccio@uniparthenope.it)
- Raffaele Montella (raffaele.montella@uniparthenope.it) — ORCID: 0000-0002-4767-2045

Affiliation: Department of Science and Technology, University of Naples "Parthenope", Naples, Italy

Funding: Hi-WeFAI project ("High-performance computing Weather nowcasting with Federated Artificial Intelligence"), National Center ICSC – "National Center for HPC, Big Data and Quantum Computing" – Cascade Call Spoke 9 – CUP E63C22000980007 ID CODE CN_00000013.

GitHub: https://github.com/Attilio-Di-Vicino/FederatedWeatherCC

---

## 5. Paper Structure Requirements

Write a complete IEEE-format paper with the following sections:

### Title
Something that captures the evolution: expanded dataset, improved preprocessing, warm-start FL. Suggested: "Scalable Federated Learning for Multi-Station Weather Forecasting: An Extended Study with Warm-Start Initialisation and Improved Data Engineering"

### Abstract (~250 words)
Cover: motivation, what is new vs previous paper (8 stations, 46k rows, warm-start FL, improved preprocessing), key results (centralised Crossformer MAE=0.1227 Skill=0.758; federated best MAE=0.1808 Skill=0.321 at round 1), conclusion about privacy-performance tradeoff.

### 1. Introduction
- Motivate the work as a direct extension of the previous paper
- Highlight the three main contributions: (1) dataset expansion to 8 stations and 17 months, (2) rigorous preprocessing pipeline with quality-based feature selection, (3) warm-start FL methodology
- Briefly state results

### 2. Related Work
- Cite all references from the previous paper that are still relevant
- Add discussion of warm-start / transfer learning in FL context
- Discuss non-IID data challenges in federated time series
- Reference the previous paper explicitly as the direct predecessor

### 3. Materials and Methods
#### 3.1 Dataset and Data Collection
- 8 stations, University of Naples "Parthenope"
- Two data structures (hierarchical directory layouts, merged)
- Raw sub-minute sampling, 4.37M rows before resampling
- Date range Jan 2025 – Jun 2026

#### 3.2 Preprocessing Pipeline
Detail all 6 steps listed in section 2.2 above. Include a table of feature selection decisions with NaN percentages.

#### 3.3 Dataset Split Strategy
Explain the per-station temporal split and why global sorting alone is insufficient for multi-station interleaved data.

#### 3.4 Infrastructure
- GPU: NVIDIA RTX 4080 16GB
- PyTorch with AMP
- Flower (FedProx)
- Compare to previous Azure Hadoop infrastructure

### 4. Model Architecture
- Transformer and Crossformer descriptions (same as before, reference previous paper)
- Updated hyperparameters table: input_dim=8, output_window=2, d_model=64, etc.
- Early stopping and LR scheduler details

### 5. Federated Learning Methodology
#### 5.1 FedProx Setup
- proximal_mu=0.01, rounds=20, local_epochs=5
#### 5.2 Warm-Start Initialisation (KEY CONTRIBUTION)
- Explain why random init fails (standalone client test: before training MAE=1.65 Skill=-19, after 10 epochs MAE=0.47 Skill=-0.73)
- Explain warm-start: load centralised checkpoint as initial global parameters
- Justify: FL rounds serve local adaptation, not global learning from scratch

### 6. Experimental Results
- Table: centralised results (both models)
- Table: federated results per round (both models) — show rounds 1, 5, 10, 15, 20
- Table: comparison with previous paper results
- Discussion of why Crossformer federated degrades over rounds (non-IID client drift)
- Discussion of Transformer FL failure

### 7. Discussion
- Privacy-performance tradeoff: centralised Skill=0.758 → federated Skill=0.321 (Crossformer)
- Non-IID data heterogeneity as main challenge
- Warm-start as a practical solution that makes FL feasible
- Limitations: 17 months still short, seasonal distribution shift in test set, FL divergence over rounds
- Honest framing of negative Transformer FL result

### 8. Conclusions and Future Work
- Summarise contributions
- Future work: differential privacy, personalised FL (per-station fine-tuning), longer dataset, more stations, comparison with FedAvg

---

## 6. Writing Instructions

- Use IEEE two-column conference format (IEEEtran class)
- All LaTeX, complete and compilable
- Include all tables formatted as IEEE tables
- Do not fabricate any numbers — use only the results provided in section 3
- Be honest about limitations and negative results (FL degrading over rounds is a valid finding)
- Cite the previous paper (CEUR-WS Vol-4124 paper51) explicitly and frequently as the direct predecessor
- The tone should position this as a rigorous engineering and systems paper, not just an application paper
- Length: target 6-8 IEEE pages
- Do NOT include the Declaration on Generative AI (the previous paper had this; leave it out of the new one or update as appropriate)

---

## 7. Key References to Include

All references from the previous paper remain valid. Add:
- The previous paper itself as [X]: Di Vicino et al., "Federated Learning for Distributed Weather Forecasting...", CEUR-WS Vol-4124, BigHPC2025/ITADATA2025, 2025
- Warm-start / transfer learning in FL: relevant papers on FedProx, and any relevant transfer learning in FL papers
- Non-IID federated learning challenges: standard references
- The Flower framework paper if available