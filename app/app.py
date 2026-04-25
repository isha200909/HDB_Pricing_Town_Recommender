import json
import datetime
import warnings
import numpy as np
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import joblib

_TOWN_DESCRIPTIONS = {
    "ANG MO KIO":      "Mature estate · Central-North",
    "BEDOK":           "Mature estate · East",
    "BISHAN":          "Mature estate · Central",
    "BUKIT BATOK":     "Mature estate · West",
    "BUKIT MERAH":     "Mature estate · Central-South",
    "BUKIT PANJANG":   "Non-mature · West",
    "BUKIT TIMAH":     "Mature estate · Central",
    "CENTRAL AREA":    "City centre · Central",
    "CHOA CHU KANG":   "Non-mature · West",
    "CLEMENTI":        "Mature estate · West",
    "GEYLANG":         "Mature estate · Central-East",
    "HOUGANG":         "Mature estate · North-East",
    "JURONG EAST":     "Non-mature · West",
    "JURONG WEST":     "Non-mature · West",
    "KALLANG/WHAMPOA": "Mature estate · Central",
    "MARINE PARADE":   "Mature estate · East",
    "PASIR RIS":       "Non-mature · East",
    "PUNGGOL":         "Non-mature · North-East",
    "QUEENSTOWN":      "Mature estate · Central",
    "SEMBAWANG":       "Non-mature · North",
    "SENGKANG":        "Non-mature · North-East",
    "SERANGOON":       "Mature estate · North-East",
    "TAMPINES":        "Non-mature · East",
    "TOA PAYOH":       "Mature estate · Central",
    "WOODLANDS":       "Non-mature · North",
    "YISHUN":          "Non-mature · North",
}

_REGION_TOWNS: dict = {
    "Central":    {"BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL AREA",
                   "GEYLANG", "KALLANG/WHAMPOA", "QUEENSTOWN", "TOA PAYOH"},
    "East":       {"BEDOK", "MARINE PARADE", "PASIR RIS", "TAMPINES"},
    "North-East": {"ANG MO KIO", "HOUGANG", "PUNGGOL", "SENGKANG", "SERANGOON"},
    "West":       {"BUKIT BATOK", "BUKIT PANJANG", "CHOA CHU KANG", "CLEMENTI",
                   "JURONG EAST", "JURONG WEST"},
    "North":      {"SEMBAWANG", "WOODLANDS", "YISHUN"},
}

_HERE = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)

_MODEL_DIR = _HERE / "models"

# ---------------------------------------------------------------------------
# Load classification model artefacts at startup
# ---------------------------------------------------------------------------
try:
    _lgbm_clf   = joblib.load(_MODEL_DIR / "lgbm_classifier.joblib")
    _clf_scaler = joblib.load(_MODEL_DIR / "scaler_classifier.joblib")
    with open(_MODEL_DIR / "cluster_labels.json") as _f:
        _CLUSTER_LABELS = json.load(_f)          # str(cluster_id) → name
    with open(_MODEL_DIR / "town_cluster_map.json") as _f:
        _TOWN_CLUSTER_MAP = json.load(_f)        # town → cluster_id
    with open(_MODEL_DIR / "classifier_feature_columns.json") as _f:
        _CLF_FEATURE_COLS = json.load(_f)        # ordered feature list
    # Reverse map: cluster_id (str) → sorted list of towns
    _CLUSTER_TOWNS: dict = {}
    for _town, _cid in _TOWN_CLUSTER_MAP.items():
        _CLUSTER_TOWNS.setdefault(str(_cid), []).append(_town)
    _CLUSTER_TOWNS = {k: sorted(v) for k, v in _CLUSTER_TOWNS.items()}
    # Pre-scale town profiles for inference-time similarity scoring
    with open(_MODEL_DIR / "town_profiles.json") as _f:
        _raw_profiles = json.load(_f)            # town → {feature: median_value}
    _TOWN_PROFILE_SCALED: dict = {}
    for _t, _vals in _raw_profiles.items():
        _vec = np.array([[_vals[c] for c in _CLF_FEATURE_COLS]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _TOWN_PROFILE_SCALED[_t] = _clf_scaler.transform(_vec)[0]
    _clf_ready = True
except FileNotFoundError:
    _clf_ready = False

# ---------------------------------------------------------------------------
# Load regression model artefacts at startup
# ---------------------------------------------------------------------------
try:
    _lgbm_reg = joblib.load(_MODEL_DIR / "lgbm_regressor.joblib")
    with open(_MODEL_DIR / "feature_columns.json") as _f:
        _FEATURE_COLS = json.load(_f)
    with open(_MODEL_DIR / "feature_medians.json") as _f:
        _FEATURE_MEDIANS = json.load(_f)
    # Label encoder classes (saved by export cell in Regression_Models_Comparison.ipynb)
    # Maps categorical column name → list of sorted unique class labels (same order as LabelEncoder)
    _label_classes_path = _MODEL_DIR / "label_classes.json"
    if _label_classes_path.exists():
        with open(_label_classes_path) as _f:
            _LABEL_CLASSES = json.load(_f)
    else:
        _LABEL_CLASSES = {}
    _reg_ready = True
except FileNotFoundError:
    _reg_ready = False


@app.route("/")
def index():
    return render_template("index.html", active_page="estimator")


@app.route("/recommender")
def recommender_page():
    return render_template("recommender.html", active_page="recommender")


# ---------------------------------------------------------------------------
# Liveability index — normalisation constants (from dataset describe() stats)
# and helper functions that replicate EDA_sprint.ipynb build_liveability()
# ---------------------------------------------------------------------------
_LIVE_NORM = {
    "mrt_nearest_distance":    (21.97,   3544.5),
    "Hawker_Nearest_Distance": (1.87,    4907.0),
    "hawker_food_stalls":      (0.0,      226.0),
    "Mall_Nearest_Distance":   (0.0,     5000.0),
    "Mall_Within_2km":         (0.0,      43.0),
    "sec_sch_nearest_dist":    (38.91,   3639.0),
}

def _inv_norm(val, col_min, col_max):
    """Inverted min-max: closer distance → higher score. Clips to [0, 1]."""
    return max(0.0, min(1.0, (col_max - float(val)) / (col_max - col_min)))

def _fwd_norm(val, col_min, col_max):
    """Forward min-max: higher value → higher score. Clips to [0, 1]."""
    if col_max == col_min:
        return 0.0
    return max(0.0, min(1.0, (float(val) - col_min) / (col_max - col_min)))


@app.route("/predict", methods=["POST"])
def predict():
    if not _reg_ready:
        msg = "Model not loaded — run the export cell in Regression_Models_Comparison.ipynb first."
        if request.headers.get("Accept") == "application/json":
            return jsonify({"price": "—", "price_raw": 0, "price_low": "—", "price_high": "—", "used_inputs": [], "price_note": msg})
        return render_template("index.html", active_tab="estimator", price="—", price_note=msg)

    prediction = 0
    try:
        floor_area      = float(request.form.get("floor_area_sqm") or _FEATURE_MEDIANS.get("floor_area_sqm", 90))
        mid_storey      = float(request.form.get("storey") or _FEATURE_MEDIANS.get("mid_storey", 8))
        remaining_lease = float(request.form.get("remaining_lease") or _FEATURE_MEDIANS.get("remaining_lease", 69))
        cbd_dist        = float(request.form.get("cbd_distance") or _FEATURE_MEDIANS.get("cbd_distance", 12.9))
        mrt_dist      = float(request.form.get("mrt_distance")    or 625)
        mrt_inter     = 1.0 if request.form.get("mrt_interchange") == "1" else 0.0
        hawker_dist   = float(request.form.get("hawker_distance") or 792)
        hawker_stalls = float(request.form.get("hawker_stalls")   or 43)
        mall_dist     = float(request.form.get("mall_distance")   or 613)
        mall_count    = float(request.form.get("mall_within_2km") or 5)
        sec_dist      = float(request.form.get("sec_distance")    or 459)
        sec_qual      = 1.0 if request.form.get("sec_quality") == "1" else 0.0
        pri_dist      = float(request.form.get("pri_distance")    or 361)
        pri_affil     = request.form.get("pri_school_type") in ("affiliated", "branded")
        pri_branded   = request.form.get("pri_school_type") == "branded"

        _mrt_prox    = _inv_norm(mrt_dist,      *_LIVE_NORM["mrt_nearest_distance"])
        _live_mrt    = 0.7 * _mrt_prox + 0.3 * mrt_inter
        _hawk_prox   = _inv_norm(hawker_dist,   *_LIVE_NORM["Hawker_Nearest_Distance"])
        _hawk_stalls = _fwd_norm(hawker_stalls, *_LIVE_NORM["hawker_food_stalls"])
        _live_hawker = 0.6 * _hawk_prox + 0.4 * _hawk_stalls
        _mall_prox   = _inv_norm(mall_dist,     *_LIVE_NORM["Mall_Nearest_Distance"])
        _mall_cnt    = _fwd_norm(mall_count,    *_LIVE_NORM["Mall_Within_2km"])
        _live_mall   = 0.5 * _mall_prox + 0.5 * _mall_cnt
        _sec_prox    = _inv_norm(sec_dist,      *_LIVE_NORM["sec_sch_nearest_dist"])
        _live_sec    = 0.5 * _sec_prox + 0.5 * sec_qual
        _dw          = 10 if pri_dist <= 1000 else (5 if pri_dist <= 2000 else 1)
        _pm          = 2.0 if pri_branded else (1.5 if pri_affil else 1.0)
        _live_pri    = (_dw * _pm - 1.0) / 19.0
        liveability  = 0.25*_live_mrt + 0.20*_live_pri + 0.20*_live_sec + 0.20*_live_mall + 0.15*_live_hawker
        flat_type  = request.form.get("flat_type", "")
        flat_model = request.form.get("flat_model", "")
        town       = request.form.get("town", "")

        current_year = datetime.datetime.now().year

        # Start with dataset medians as fallback for every feature
        row = {col: _FEATURE_MEDIANS.get(col, 0.0) for col in _FEATURE_COLS}

        # --- Numeric features provided directly by the user ---
        row["floor_area_sqm"]  = floor_area
        row["mid_storey"]      = mid_storey
        row["Tranc_Year"]      = current_year
        row["remaining_lease"] = remaining_lease
        row["cbd_distance"]    = cbd_dist
        row["liveability_index"] = liveability
        if "lease_commence_date" in row:
            row["lease_commence_date"] = current_year - (99 - int(remaining_lease))

        # --- Derived from town selection ---
        _mature_estates = {
            "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
            "CENTRAL AREA", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA",
            "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
            "TAMPINES", "TOA PAYOH",
        }
        if town and "mature_estate" in row:
            row["mature_estate"] = 1.0 if town.upper() in _mature_estates else 0.0

        # --- Label-encoded categoricals (flat_type, flat_model) ---
        # LabelEncoder assigns the sorted-index of each class label.
        def _label_encode(col, value):
            classes = _LABEL_CLASSES.get(col, [])
            if value and value in classes:
                return float(classes.index(value))
            return _FEATURE_MEDIANS.get(col, 0.0)

        if flat_type and "flat_type" in row:
            row["flat_type"] = _label_encode("flat_type", flat_type)
        if flat_model and "flat_model" in row:
            row["flat_model"] = _label_encode("flat_model", flat_model)
            # Derive DBSS flag from flat_model
            if "is_dbss" in row:
                row["is_dbss"] = 1.0 if "DBSS" in flat_model.upper() else 0.0

        features = np.array([[row[col] for col in _FEATURE_COLS]])
        prediction = _lgbm_reg.predict(features)[0]

        price_str      = f"${prediction:,.0f}"
        price_low_str  = f"${prediction * 0.90:,.0f}"
        price_high_str = f"${prediction * 1.10:,.0f}"
        note = "Estimate based on available inputs; other features use dataset medians."

        used_inputs = []
        if request.form.get("flat_type"):        used_inputs.append(request.form.get("flat_type").title())
        if request.form.get("flat_model"):       used_inputs.append(request.form.get("flat_model").title())
        if request.form.get("town"):             used_inputs.append(request.form.get("town").title())
        if request.form.get("floor_area_sqm"):   used_inputs.append(f"{request.form.get('floor_area_sqm')} sqm")
        if request.form.get("storey"):           used_inputs.append(f"Storey {request.form.get('storey')}")
        if request.form.get("remaining_lease"):  used_inputs.append(f"{int(remaining_lease)}yr lease left")
        if request.form.get("cbd_distance"):     used_inputs.append(f"{cbd_dist:.1f} km to CBD")
        _any_liv = any(request.form.get(k) for k in (
            "mrt_distance", "mrt_interchange", "hawker_distance", "hawker_stalls",
            "mall_distance", "mall_within_2km", "sec_distance", "sec_quality",
            "pri_distance", "pri_school_type"))
        if _any_liv: used_inputs.append(f"Liveability {liveability:.2f}")

    except Exception as e:
        price_str = price_low_str = price_high_str = "—"
        used_inputs = []
        note = f"Error: {e}"

    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "price": price_str,
            "price_raw": int(float(prediction)),
            "price_low": price_low_str,
            "price_high": price_high_str,
            "used_inputs": used_inputs,
            "price_note": note,
        })
    return render_template(
        "index.html",
        active_tab="estimator",
        price=price_str,
        price_raw=int(float(prediction)),
        price_low=price_low_str,
        price_high=price_high_str,
        used_inputs=used_inputs,
        price_note=note,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    if not _clf_ready:
        msg = "Model not loaded — run the export cell in recommender_kmeans.ipynb first."
        if request.headers.get("Accept") == "application/json":
            return jsonify({"rec_cluster": "—", "rec_towns": [], "error": msg})
        return render_template("index.html", active_tab="recommender", recommendation=msg)

    pred_cluster_name  = None
    cluster_confidence = 0.0
    rec_towns = []
    try:
        # ── 1. Read form inputs ───────────────────────────────────────────
        resale_price   = float(request.form.get("resale_price")    or 500000)
        floor_area_sqm = float(request.form.get("floor_area_sqm")  or 93)
        hdb_age        = float(request.form.get("hdb_age")         or 20)
        distance_cbd   = float(request.form.get("cbd_distance_km") or 12)
        max_floor_lvl  = float(request.form.get("max_floor_lvl")   or 15)
        storey_ratio    = float(request.form.get("storey_ratio")    or 0.5)
        planning_region = request.form.get("planning_region", "").strip()

        # ── 2. Map toggles → proxy distances (metres) ────────────────────
        mrt_dist    = 500.0  if request.form.get("near_mrt")    == "1" else 1500.0
        hawker_dist = 300.0  if request.form.get("near_hawker") == "1" else 1000.0
        mall_dist   = 500.0  if request.form.get("near_mall")   == "1" else 1500.0
        school_dist = 500.0  if request.form.get("near_school") == "1" else 1500.0

        # ── 3. Compute derived features ───────────────────────────────────
        # cbd_distance_band: bin continuous distance into 4 ordinal bands
        if distance_cbd <= 5:
            cbd_distance_band = 1.0
        elif distance_cbd <= 10:
            cbd_distance_band = 2.0
        elif distance_cbd <= 15:
            cbd_distance_band = 3.0
        else:
            cbd_distance_band = 4.0

        # estate_height_modernity: tall-new vs tall-old signal
        estate_height_modernity = max_floor_lvl / (hdb_age + 1)

        # amenity_cluster tiers: count of amenity types within each radius
        amenity_500m = (int(mrt_dist < 500) + int(mall_dist < 500)
                        + int(hawker_dist < 500) + int(school_dist < 500))
        amenity_1km  = (int(mrt_dist < 1000) + int(mall_dist < 1000)
                        + int(hawker_dist < 1000) + int(school_dist < 1000))
        amenity_2km  = (int(mrt_dist < 2000) + int(mall_dist < 2000)
                        + int(hawker_dist < 2000) + int(school_dist < 2000))

        # block_diversity: Shannon entropy of flat-type mix — not a form input.
        # Dataset median (~0.8) used as inference-time default.
        block_diversity = 0.8

        # ── 4. Build feature array in exact training order ────────────────
        feature_map = {
            "resale_price":             resale_price,
            "floor_area_sqm":           floor_area_sqm,
            "block_diversity":          block_diversity,
            "cbd_distance_band":        cbd_distance_band,
            "mrt_nearest_distance":     mrt_dist,
            "Hawker_Nearest_Distance":  hawker_dist,
            "Mall_Nearest_Distance":    mall_dist,
            "pri_sch_nearest_distance": school_dist,
            "storey_ratio":             storey_ratio,
            "estate_height_modernity":  estate_height_modernity,
            "amenity_cluster_500m":     float(amenity_500m),
            "amenity_cluster_1km":      float(amenity_1km),
            "amenity_cluster_2km":      float(amenity_2km),
        }
        features = np.array([[feature_map[col] for col in _CLF_FEATURE_COLS]])

        # ── 5. Scale → predict cluster + confidence ───────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_scaled    = _clf_scaler.transform(features)
            proba              = _lgbm_clf.predict_proba(features_scaled)[0]
        pred_cluster_idx   = int(np.argmax(proba))
        cluster_confidence = float(proba[pred_cluster_idx])

        pred_cluster_name = _CLUSTER_LABELS[str(pred_cluster_idx)]
        cluster_towns     = _CLUSTER_TOWNS.get(str(pred_cluster_idx), [])

        # Always score every town in the selected region (cluster used only for confidence)
        if planning_region and planning_region in _REGION_TOWNS:
            cluster_towns = sorted(
                t for t in _REGION_TOWNS[planning_region] if t in _TOWN_PROFILE_SCALED
            )

        # ── 6. Score each town by L2 distance in scaled feature space ────
        buyer_vec = features_scaled[0]
        scored = []
        for town in cluster_towns:
            profile = _TOWN_PROFILE_SCALED.get(town)
            if profile is None:
                scored.append((town, 9999.0))
                continue
            dist = float(np.linalg.norm(buyer_vec - profile))
            scored.append((town, dist))

        if scored:
            raw        = np.array([s[1] for s in scored])
            exp_scores = np.exp(-raw)                        # closer → higher
            norm_scores = exp_scores / exp_scores.max()     # best town = 1.0
            final      = [round(float(s) * cluster_confidence * 100)
                          for s in norm_scores]
            rec_towns  = [
                {"name": t, "score": sc}
                for (t, _), sc in sorted(
                    zip(scored, final), key=lambda x: x[1], reverse=True
                )
            ]
        else:
            rec_towns = []

        result = f"Recommended cluster: {pred_cluster_name}"
    except Exception as e:
        result = f"Error: {e}"

    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "rec_cluster":    pred_cluster_name or "—",
            "rec_confidence": round(cluster_confidence * 100) if pred_cluster_name else 0,
            "rec_towns":      rec_towns,
        })
    return render_template(
        "index.html",
        active_tab="recommender",
        recommendation=result,
        rec_cluster=pred_cluster_name or "—",
        rec_towns=rec_towns,
    )


if __name__ == "__main__":
    import os
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, use_reloader=debug)
