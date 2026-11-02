
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd


# 1. CONFIGURAÇÃO DOS CAMINHOS Configuração dos caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "titanic"

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "titanic"

QUALITY_PATH = SILVER_PATH / "data_quality"

TRAIN_PATH = BRONZE_PATH / "train.csv"
TEST_PATH = BRONZE_PATH / "test.csv"


# 2. CARREGAMENTO DOS DADOS Carregamento dos dados
def load_data():

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    return {
        "train": df_train,
        "test": df_test
    }


# 3. Regras de Qualidade
def check_required_columns(df):

    required_columns = [
        "PassengerId",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "Survived"
    ]

    # A coluna Survived não existe no test.csv
    required_columns = [
        column for column in required_columns
        if column in df.columns
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    return len(missing_columns) == 0, missing_columns


def check_null_identifiers(df):

    null_count = int(df["PassengerId"].isna().sum())

    return null_count == 0, null_count


def check_duplicate_identifiers(df):

    duplicate_count = int(
        df["PassengerId"].duplicated().sum()
    )

    return duplicate_count == 0, duplicate_count


def check_valid_passenger_id(df):

    invalid_count = int(
        (df["PassengerId"] <= 0).sum()
    )

    return invalid_count == 0, invalid_count


def check_valid_sex(df):

    valid_values = ["male", "female"]

    invalid_count = int(
        (~df["Sex"].isin(valid_values)).sum()
    )

    return invalid_count == 0, invalid_count


def check_valid_pclass(df):

    valid_values = [1, 2, 3]

    invalid_count = int(
        (~df["Pclass"].isin(valid_values)).sum()
    )

    return invalid_count == 0, invalid_count


def check_valid_survived(df):

    if "Survived" not in df.columns:
        return None, None

    valid_values = [0, 1]

    invalid_count = int(
        (~df["Survived"].isin(valid_values)).sum()
    )

    return invalid_count == 0, invalid_count


# ============================================================
# 4. EXECUTAR AS REGRAS
# ============================================================

def validate_dataset(df, dataset_name):

    results = []

    checks = [
        ("required_columns", check_required_columns),
        ("null_identifiers", check_null_identifiers),
        ("duplicate_identifiers", check_duplicate_identifiers),
        ("valid_passenger_id", check_valid_passenger_id),
        ("valid_sex", check_valid_sex),
        ("valid_pclass", check_valid_pclass),
        ("valid_survived", check_valid_survived)
    ]

    for rule_name, check_function in checks:

        passed, details = check_function(df)

        if passed is None:
            status = "NOT_APPLICABLE"
        else:
            status = "PASS" if passed else "FAIL"

        results.append({

            "dataset": dataset_name,
            "rule": rule_name,
            "status": status,
            "details": json.dumps(
                details,
                ensure_ascii=False
            )

        })

    return results



# 5. SALVAR RELATÓRIO Salvar Relatório
def save_quality_report(results):

    QUALITY_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    df_quality = pd.DataFrame(results)

    timestamp = datetime.now(
        timezone.utc
    )

    df_quality["processed_at"] = timestamp

    file_timestamp = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        QUALITY_PATH /
        f"data_quality_{file_timestamp}.parquet"
    )

    df_quality.to_parquet(
        output_path,
        index=False
    )

    print(f"\nRelatório salvo em: {output_path}")

    return df_quality


# 6. Execução
def main():

    print("\nIniciando validação de qualidade...\n")

    datasets = load_data()

    all_results = []

    for dataset_name, df in datasets.items():

        results = validate_dataset(
            df,
            dataset_name
        )

        all_results.extend(results)

    df_quality = save_quality_report(all_results)

    print("\nResultado das validações:")
    print(df_quality.to_string(index=False))

    print("\nValidação de qualidade finalizada!")


if __name__ == "__main__":
    main()
