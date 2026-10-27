"""
Lab 04 configuration: paths, schema, and pipeline settings.

Everything the pipeline depends on lives here so that a run is fully
described by this file. The values are also written to
results/config.yaml together with the processed-data hash.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"

RAW_FILE = RAW_DIR / "adult_raw.csv"

# ============================================================
# DATASET
# ============================================================

DATASET = {
    "name": "adult",
    "version": 2,
    "source": "OpenML (dataset 1590) via sklearn.datasets.fetch_openml",
    "expected_rows": 48842,
    "expected_columns": 15,
}

TARGET = "income"
POSITIVE_LABEL = ">50K"
MISSING_TOKEN = "?"

# ============================================================
# SCHEMA (types, allowed categories, ranges)
# ============================================================

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]

SCHEMA_TYPES = {col: "int" for col in NUMERIC_COLUMNS}
SCHEMA_TYPES.update({col: "category" for col in CATEGORICAL_COLUMNS})
SCHEMA_TYPES[TARGET] = "category"

# Inclusive ranges from the lab sheet.
RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

# Allowed values from the UCI adult.names data dictionary.
# '?' is allowed everywhere because it is the dataset's missing marker.
ALLOWED_CATEGORIES = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    ],
    "education": [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black",
    ],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
        "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
        "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
        "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
        "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru",
        "Hong", "Holand-Netherlands",
    ],
    "income": ["<=50K", ">50K"],
}

# education and education-num encode the same thing in the data dictionary.
EDUCATION_TO_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}

# ============================================================
# PIPELINE SETTINGS
# ============================================================

SEED = 42
TEST_SIZE = 0.30
CLIP_PERCENTILES = (1, 99)
LABEL_AUDIT_SAMPLE_SIZE = 50
CV_FOLDS = 5
LOGREG_MAX_ITER = 2000
