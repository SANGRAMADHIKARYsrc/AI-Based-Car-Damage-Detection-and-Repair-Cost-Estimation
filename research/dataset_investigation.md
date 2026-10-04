# Dataset Investigation Report (Pre-Phase 2)

**Date:** 2026-09-26
**Purpose:** Document each candidate dataset's properties, license, and suitability BEFORE downloading.

---

## Damage Detection Datasets

### 1. CarDD
- **Source:** https://cardd-ustc.github.io/
- **Paper:** Wang et al. (2023), IEEE TITS
- **Images:** 4,000 high-resolution
- **Annotations:** 9,000+ instances
- **Classes:** 6 — dent, scratch, crack, glass shatter, tire flat, lamp broken
- **Annotation format:** Bounding boxes + pixel-level masks
- **Tasks supported:** Object detection, instance segmentation, SOD
- **Average resolution:** ~684K pixels per image
- **License:** Requires license agreement via project website
- **Component labels:** NO
- **Severity labels:** NO
- **Cost labels:** NO
- **Suitability:** PRIMARY detection dataset. Well-established benchmark. YOLO annotations available via Kaggle.

### 2. CarDD YOLO Annotations
- **Source:** https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels
- **Description:** Pre-converted YOLO-format annotations for CarDD images
- **Format:** YOLO txt format (class x_center y_center width height)
- **Benefit:** Saves annotation conversion work for YOLO training
- **License:** Check Kaggle page — likely follows CarDD license
- **Note:** Must verify annotation quality and completeness against original CarDD

### 3. VehiDE
- **Source:** https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection
- **Paper:** Huynh et al. (2023/2024), Taylor & Francis
- **Images:** 13,945 high-resolution
- **Annotations:** 32,000+
- **Classes:** 8 damage categories
- **Tasks supported:** Classification, object detection, instance segmentation, SOD
- **License:** Check Kaggle page
- **Component labels:** NO (based on paper description)
- **Severity labels:** NO
- **Cost labels:** NO
- **Suitability:** SECONDARY detection dataset. Useful for cross-dataset generalization (Phase 11). Larger than CarDD.

### 4. Car Damage Severity Dataset
- **Source:** https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset
- **Description:** Classification dataset with severity-labeled images
- **Expected classes:** Minor, Moderate, Severe (needs verification)
- **Format:** Likely folder-based classification (image folders per class)
- **License:** Check Kaggle page
- **Component labels:** UNKNOWN — needs inspection
- **Damage type labels:** UNKNOWN — needs inspection
- **Cost labels:** NO
- **Suitability:** Potential source for severity classification training (Phase 7). Must verify label quality.

---

## Cost-Related Datasets

### 5. Copart Dataset
- **Source:** https://github.com/rebrowser/copart-dataset
- **Description:** Auction/salvage vehicle data from Copart
- **Expected fields:** Vehicle make, model, year, damage description, sale price
- **WARNING:** Sale price ≠ repair cost. Salvage auction price reflects vehicle residual value, NOT repair cost.
- **License:** Check GitHub repository
- **Suitability:** LIMITED. May provide vehicle metadata but NOT direct repair cost data. Must NOT conflate auction price with repair cost.

### 6. Car Parts Price Dataset
- **Source:** https://www.kaggle.com/datasets/shorooq77/car-parts-price-estimation/versions/1
- **Description:** Parts pricing data
- **Expected fields:** Part name, price, possibly make/model
- **License:** Check Kaggle page
- **Suitability:** USEFUL for parts_cost component in repair cost estimation. Must verify currency, region, and recency.

### 7. Motor Vehicle Repair Dataset
- **Source:** https://www.kaggle.com/datasets/aryan208/motor-vehicle-repair-and-towing-dataset
- **Description:** Repair and towing records
- **Expected fields:** Repair type, cost, vehicle info
- **License:** Check Kaggle page
- **WARNING:** May include towing costs, which are NOT repair costs. Must separate.
- **Suitability:** POTENTIALLY USEFUL. Needs careful inspection to extract only repair-relevant records.

### 8. Vehicle Maintenance Record
- **Source:** https://www.kaggle.com/navins7/vehicle-maintenance-record
- **Description:** Maintenance records
- **WARNING:** Maintenance cost (oil change, tire rotation, scheduled service) ≠ accident repair cost. These are FUNDAMENTALLY DIFFERENT.
- **License:** Check Kaggle page
- **Suitability:** LIKELY NOT SUITABLE for this project. Maintenance ≠ collision repair. Document and exclude with clear reasoning.

---

## Class Mapping Considerations

CarDD classes (6): dent, scratch, crack, glass_shatter, tire_flat, lamp_broken
VehiDE classes (8): needs verification — likely overlaps but may use different names

**ACTION NEEDED:** When datasets are downloaded, create explicit `data/class_mapping.yaml` mapping between CarDD and VehiDE class names.

---

## Key Decisions for Phase 2

1. **Primary detection dataset:** CarDD (with YOLO annotations from Kaggle)
2. **Cross-dataset evaluation:** VehiDE
3. **Severity training:** Car Damage Severity Dataset (if labels are valid)
4. **Cost data:** Car Parts Price + Motor Vehicle Repair (if applicable)
5. **EXCLUDED:** Vehicle Maintenance Record (maintenance ≠ repair)
6. **CAUTION:** Copart (auction price ≠ repair cost)

---

## Next Step

Download datasets, perform Phase 2 analysis (counts, class distributions, resolution stats, corruption checks, duplicate detection).
