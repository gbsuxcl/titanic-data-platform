
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# Caminhos do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "titanic"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "titanic"
CLEANED_PATH = SILVER_PATH / "cleaned"

TRAIN_PATH = BRONZE_PATH / "train.csv"
TEST_PATH = BRONZE_PATH / "test.csv"


def load_data():
    """Carrega os dados originais da camada Bronze."""

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return {
        "train": train,
        "test": test
    }


def clean_text_columns(df):
    """Remove espaços extras e transforma textos vazios em nulos."""

    df = df.copy()

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

        df[column] = df[column].replace("", "VALOR_IMPLEMENTADO")

    return df


def replace_null_values(df):
    """Substitui valores nulos das colunas numéricas por -999999."""

    df = df.copy()

    numeric_columns = df.select_dtypes(include="number").columns

    df[numeric_columns] = df[numeric_columns].fillna(-999999)

    return df


def remove_exact_duplicates(df):
    """Remove registros completamente duplicados."""

    df = df.copy()

    total_before = len(df)

    df = df.drop_duplicates()

    total_after = len(df)

    removed = total_before - total_after

    return df, removed


def clean_dataset(df, dataset_name):
    """Executa a limpeza estrutural do dataset."""

    df = df.copy()

    # Limpeza de textos
    df = clean_text_columns(df)

    # Remoção de duplicidades exatas
    df, duplicates_removed = remove_exact_duplicates(df)

    # Substituição de valores nulos
    df = replace_null_values(df)

    print(f"\nDataset: {dataset_name}")
    print(f"Registros após limpeza: {len(df)}")
    print(f"Duplicidades removidas: {duplicates_removed}")

    return df



def save_data(data):
    """Salva os dados tratados em formato Parquet."""

    CLEANED_PATH.mkdir(parents=True, exist_ok=True)

    for dataset_name, df in data.items():

        output_path = CLEANED_PATH / f"{dataset_name}_clean.parquet"

        df.to_parquet(output_path, index=False)

        print(f"Arquivo salvo: {output_path}")


def main():

    print("Iniciando limpeza dos dados...")

    data = load_data()

    cleaned_data = {}

    for dataset_name, df in data.items():

        cleaned_data[dataset_name] = clean_dataset(
            df,
            dataset_name
        )

    save_data(cleaned_data)

    processed_at = datetime.now(timezone.utc)

    print("\nLimpeza finalizada!")
    print(f"Processado em: {processed_at.isoformat()}")


if __name__ == "__main__":
    main()
