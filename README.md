[README.md](https://github.com/user-attachments/files/34539920/README.md)
# 🐦 AI-Based Bird Biodiversity Monitoring System: Bengaluru

A deep-learning and data-analysis project that identifies common Bengaluru bird species from photographs and analyses where, when and in which habitats they are recorded.

**Live dashboard:** [paste your .streamlit.app link here]

---

## What it does

1. **Identify:** upload a bird photo; a CNN predicts the species and a confidence score.
2. **Record:** save the observation with date, location, season, habitat and confidence.
3. **Analyse:** compare recorded observations across seasons, years, habitats and locations.
4. **Flag changes:** highlight notable shifts in recorded observations between two time periods, adjusted for observation effort.
5. **Explore:** view everything in an interactive Streamlit dashboard.

## Species covered

| Species | Scientific name |
|---|---|
| Common Myna | *Acridotheres tristis* |
| Rose-ringed Parakeet | *Psittacula krameri* |
| White-throated Kingfisher | *Halcyon smyrnensis* |
| Indian Pond-Heron | *Ardeola grayii* |
| Black Drongo | *Dicrurus macrocercus* |

## How it works

| Stage | Method |
|---|---|
| Training images | Licensed, research-grade photos from [GBIF](https://www.gbif.org) (iNaturalist), with licence and photographer recorded; exact duplicates removed |
| Model | MobileNetV2 (ImageNet transfer learning) with data augmentation and class weights |
| Evaluation | Held-out test set, split **by photographer** to prevent near-duplicate leakage; accuracy, precision, recall, F1, confusion matrix |
| Occurrence data | GBIF human-observation records inside a Bengaluru bounding box, 2018 to 2025 |
| Database | SQLite (`observations`, `effort`, `species_rates`, `shift_results`, `spatial_shift`) |
| Effort adjustment | Species records ÷ all bird records in the same period, so more birders does not look like more birds |
| Shift detection | Compares 2018 to 2021 with 2022 to 2025; flags a change only when it is both large (≥ 25%) and statistically significant (p < 0.01) |
| Dashboard | Streamlit |

## Model performance

Evaluated on unseen test images:

| Metric | Result |
|---|---|
| Overall accuracy | [your final accuracy] |
| Macro F1 | [your final macro F1] |

Per-species results and the confusion matrix: [add screenshot or link].

## Repository contents

```
app.py                    Streamlit dashboard
requirements.txt          Python dependencies
bird_model.keras          Trained classifier
model_info.json           Class names and input image size
bird_observations.db      SQLite database of observations and analysis results
```

## Running the dashboard

**Online (no installation):** open the live link above.

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Limitations

- The classifier knows only five species; any other bird will be forced into one of them. Low-confidence predictions should be checked manually.
- Training photos are citizen-science images of varying quality.
- Records show where and when birds were **observed**, not true abundance. Effort adjustment reduces but does not remove observer bias.
- Habitat is assigned from the nearest reference site within 2 km, which is an approximation.
- Shift flags are screening indicators, not proof of population change. Records from the same checklist or observer are not fully independent.
- On free hosting, observations saved through the dashboard may reset when the app restarts; use the CSV download.

## Data, licences and ethics

- Occurrence records and images come from GBIF and its contributing datasets. Image licences and photographers are listed in `image_metadata.csv` (generated during dataset preparation); please retain attribution if reusing images.
- Uploaded photos in the dashboard are used for prediction only and are not stored; only the observation details are saved.

## Tools

TensorFlow / Keras · scikit-learn · pandas · Streamlit · SQLite · GBIF API · Google Colab · GitHub

## Author

[Your name] · [Your institution / course] · [Year]
