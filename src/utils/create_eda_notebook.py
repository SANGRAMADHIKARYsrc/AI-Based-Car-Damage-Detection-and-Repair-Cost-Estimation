import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 2: Exploratory Data Analysis & Quality Audit\n",
    "Comprehensive audit of CarDD YOLO dataset, Severity (data3a), VehiDE VIA polygons, and Copart auction repair cost dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "\n",
    "# Load audit results\n",
    "stats_path = Path(\"../reports/dataset_analysis_stats.json\")\n",
    "with open(stats_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    stats = json.load(f)\n",
    "\n",
    "print(\"Audit keys:\", list(stats.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. CarDD Class Distribution\n",
    "cardd = stats[\"cardd_yolo\"]\n",
    "df_classes = pd.DataFrame({\n",
    "    \"Train\": cardd[\"train\"][\"class_distribution\"],\n",
    "    \"Val\": cardd[\"val\"][\"class_distribution\"],\n",
    "    \"Test\": cardd[\"test\"][\"class_distribution\"]\n",
    "})\n",
    "df_classes[\"Total\"] = df_classes.sum(axis=1)\n",
    "df_classes[\"Percentage\"] = (df_classes[\"Total\"] / df_classes[\"Total\"].sum() * 100).round(2)\n",
    "df_classes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Severity Classification Dataset (data3a)\n",
    "df_sev = pd.DataFrame(stats[\"severity_data3a\"]).T\n",
    "df_sev"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Copart Auction Repair Cost Stats\n",
    "pd.Series(stats[\"copart_repair_cost\"][\"repair_cost_stats\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Leakage and Duplicates Audit\n",
    "pd.Series(stats[\"leakage_and_duplicates\"])"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("c:/Users/ASUS/OneDrive/Desktop/Antigravity/Capstone Project/car-damage-ai/notebooks/01_dataset_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Created notebooks/01_dataset_eda.ipynb")
