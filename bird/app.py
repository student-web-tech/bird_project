import os, json, math, sqlite3, datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Bengaluru Bird Biodiversity Monitor", page_icon="🐦", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "bird_observations.db")
MODEL_PATH = os.path.join(BASE, "bird_model.keras")
INFO_PATH = os.path.join(BASE, "model_info.json")
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
LOW_CONF = 0.60

# Same reference sites as step 3 (keep these in sync if you corrected any coordinates)
SITES = [
    ("Hebbal Lake", 13.0450, 77.5900, "Wetland"),
    ("Sankey Tank", 13.0100, 77.5710, "Wetland"),
    ("Ulsoor Lake", 12.9830, 77.6200, "Wetland"),
    ("Bellandur Lake", 12.9370, 77.6710, "Wetland"),
    ("Varthur Lake", 12.9380, 77.7400, "Wetland"),
    ("Jakkur Lake", 13.0780, 77.6080, "Wetland"),
    ("Lalbagh", 12.9507, 77.5848, "Park/Garden"),
    ("Cubbon Park", 12.9763, 77.5929, "Park/Garden"),
    ("GKVK Campus", 13.0750, 77.5800, "Open/Agricultural"),
    ("Hesaraghatta", 13.1400, 77.4900, "Open/Agricultural"),
    ("Turahalli Forest", 12.8999, 77.5100, "Scrub/Forest"),
    ("Bannerghatta NP", 12.8000, 77.5770, "Scrub/Forest"),
]
SITE_RADIUS_KM = 2.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def assign_site(lat, lon):
    best = min(SITES, key=lambda s: haversine(lat, lon, s[1], s[2]))
    if haversine(lat, lon, best[1], best[2]) <= SITE_RADIUS_KM:
        return best[0], best[3]
    return "Other (urban/residential)", "Urban"


def season_of(m):
    return ("Winter" if m in (12, 1, 2) else "Summer" if m in (3, 4, 5)
            else "Monsoon" if m in (6, 7, 8, 9) else "Post-monsoon")


@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    info = json.load(open(INFO_PATH))
    return model, info["classes"], info["img_size"]


def predict(img):
    import tensorflow as tf
    model, classes, size = load_model()
    x = np.array(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype="float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    p = model.predict(x[None], verbose=0)[0]
    order = np.argsort(p)[::-1]
    return [(classes[i], float(p[i])) for i in order]


@st.cache_data(ttl=300)
def q(sql):
    con = sqlite3.connect(DB)
    df = pd.read_sql(sql, con)
    con.close()
    return df


def save_observation(species, date, lat, lon, site, habitat, conf):
    con = sqlite3.connect(DB)
    con.execute(
        """INSERT INTO observations (source, gbif_id, species, observed_date, year, month,
           season, lat, lon, site, habitat, confidence, dataset)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("dashboard_upload", None, species, date.isoformat(), date.year, date.month,
         season_of(date.month), lat, lon, site, habitat, conf, "dashboard"))
    con.commit()
    con.close()


st.title("🐦 AI-Based Bird Biodiversity Monitoring: Bengaluru")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Identify a bird", "Biodiversity insights", "My observations", "About & limitations"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Upload a bird photo")
    up = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
        c1, c2 = st.columns(2)
        c1.image(img, use_container_width=True)
        preds = predict(img)
        top, conf = preds[0]
        c2.metric("Predicted species", top)
        c2.progress(conf, text=f"Confidence: {conf * 100:.1f}%")
        c2.write("Other possibilities:")
        for name, p in preds[1:3]:
            c2.write(f"- {name}: {p * 100:.1f}%")
        if conf < LOW_CONF:
            c2.warning("Low confidence. The photo may be unclear, or the bird may not be one of "
                       "the five species this model knows.")

        st.markdown("#### Record this observation")
        d1, d2 = st.columns(2)
        date = d1.date_input("Date observed", datetime.date.today())
        site_names = [s[0] for s in SITES] + ["Other location (enter coordinates)"]
        choice = d2.selectbox("Location", site_names)
        if choice.startswith("Other"):
            lat = d1.number_input("Latitude", value=12.97, format="%.5f")
            lon = d2.number_input("Longitude", value=77.59, format="%.5f")
            site, habitat = assign_site(lat, lon)
        else:
            _, lat, lon, habitat = next(s for s in SITES if s[0] == choice)
            site = choice
        st.caption(f"Season: **{season_of(date.month)}**  |  Habitat: **{habitat}**  |  Site: **{site}**")

        ok = True
        if conf < LOW_CONF:
            ok = st.checkbox("I have checked this identification myself")
        if st.button("Save observation", disabled=not ok):
            save_observation(top, date, lat, lon, site, habitat, conf)
            st.success("Observation saved.")

# ---------------------------------------------------------------- Tab 2
with tab2:
    rates = q("SELECT * FROM species_rates")
    obs = q("SELECT species, season, habitat, lat, lon, year FROM observations")
    species_all = sorted(rates.species.unique())
    chosen = st.multiselect("Species", species_all, default=species_all)
    if not chosen:
        st.info("Select at least one species.")
    else:
        r = rates[rates.species.isin(chosen)]
        st.caption("Rates are records per 1,000 bird records of any species, so changes in the "
                   "number of birders do not masquerade as changes in birds.")

        a, b = st.columns(2)
        seas = r.groupby(["species", "season"]).agg(n=("n", "sum"), t=("total_bird_records", "sum")).reset_index()
        seas["rate"] = seas.n / seas.t * 1000
        a.markdown("**Seasonal pattern**")
        a.bar_chart(seas.pivot(index="season", columns="species", values="rate").reindex(SEASON_ORDER))

        yr = r.groupby(["species", "year"]).agg(n=("n", "sum"), t=("total_bird_records", "sum")).reset_index()
        yr["rate"] = yr.n / yr.t * 1000
        b.markdown("**Yearly trend**")
        b.line_chart(yr.pivot(index="year", columns="species", values="rate"))

        st.markdown("**Habitat share of records**")
        h = obs[obs.species.isin(chosen)].groupby(["species", "habitat"]).size().unstack(fill_value=0)
        st.bar_chart((h.div(h.sum(axis=1), axis=0) * 100))

        st.markdown("**Where records come from**")
        m1, m2 = st.columns(2)
        sp_map = m1.selectbox("Species for map", chosen)
        season_map = m2.selectbox("Season", ["All"] + SEASON_ORDER)
        mm = obs[obs.species == sp_map]
        if season_map != "All":
            mm = mm[mm.season == season_map]
        st.map(mm[["lat", "lon"]].sample(min(2000, len(mm)), random_state=0) if len(mm) else mm[["lat", "lon"]])

        st.markdown("**Distribution-shift indicators (2018 to 2021 vs 2022 to 2025)**")
        shift = q("SELECT * FROM shift_results")
        shift = shift[shift.species.isin(chosen)][
            ["species", "season", "rate_A", "rate_B", "rel_change_pct", "flag"]]
        st.dataframe(shift, hide_index=True, use_container_width=True)
        st.markdown("**Spatial and habitat shift**")
        st.dataframe(q("SELECT * FROM spatial_shift"), hide_index=True, use_container_width=True)
        st.caption("A flag means a change in *recorded observations*. It is a prompt for further "
                   "investigation, not proof that the real bird population changed.")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Observations saved from this dashboard")
    con = sqlite3.connect(DB)
    mine = pd.read_sql(
        "SELECT observed_date, species, confidence, site, habitat, season, lat, lon "
        "FROM observations WHERE source='dashboard_upload' ORDER BY id DESC", con)
    con.close()
    if mine.empty:
        st.info("Nothing saved yet. Identify a bird in the first tab and click Save.")
    else:
        st.dataframe(mine, hide_index=True, use_container_width=True)
        st.download_button("Download as CSV", mine.to_csv(index=False), "my_observations.csv")
    st.caption("On free hosting, saved observations can be reset when the app restarts, so "
               "download your CSV regularly.")

# ---------------------------------------------------------------- Tab 4
with tab4:
    st.markdown("""
**How it works.** A MobileNetV2-based CNN, trained on licensed, research-grade photos from GBIF,
classifies five Bengaluru species. Location, date, season and habitat are stored with each
observation. Biodiversity patterns come from public GBIF occurrence records.

**Limitations**
- The classifier knows only five species. Any other bird will be forced into one of them, so check low-confidence results.
- Training photos come from citizen-science records and vary in quality; accuracy on unseen photos is the test-set figure in the project report.
- Records show where and when people *observed* birds, not true abundance. Effort adjustment reduces but does not remove this bias.
- Habitat is assigned from the nearest reference site (within 2 km), which is an approximation.
- Shift flags are screening indicators; statistical tests assume independent records, which citizen-science data only roughly satisfies.
""")
