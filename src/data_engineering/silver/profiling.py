
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# 1. Configuração dos caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "titanic"

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "titanic"

PROFILING_PATH = SILVER_PATH / "profiling"

TRAIN_PATH = BRONZE_PATH / "train.csv"
TEST_PATH = BRONZE_PATH / "test.csv"


# 2. Carregamento dos dados
def load_data():

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    return {
        "train": df_train,
        "test": df_test
    }


# 3. Inspeção da estrutura
def inspect_structure(df, dataset_name):

    print(f"\n{'=' * 50}")
    print(f"ESTRUTURA: {dataset_name}")
    print(f"{'=' * 50}")

    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nPrimeiros registros:")
    print(df.head())



# 4. Coleta das métricas de profiling
def collect_profiling_metrics(df, dataset_name, processed_at):

    total_rows = len(df)
    total_columns = len(df.columns)

    metrics = []

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        missing_percentage = (
            (missing_count / total_rows) * 100
            if total_rows > 0 else 0
        )

        metrics.append({

            "dataset": dataset_name,
            "column": column,
            "data_type": str(df[column].dtype),

            "total_rows": total_rows,
            "total_columns": total_columns,

            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage, 2
            ),

            "unique_values": int(
                df[column].nunique()
            ),

            "duplicate_rows": int(
                df.duplicated().sum()
            ),

            "processed_at": processed_at

        })

    return metrics


# 5. Salvamento das métricas de profiling
def save_profiling_dataset(metrics):

    PROFILING_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    df_profiling = pd.DataFrame(metrics)

    # Identificador temporal da execução
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_path = (
        PROFILING_PATH /
        f"profiling_{timestamp}.parquet"
    )

    df_profiling.to_parquet(
        output_path,
        index=False
    )

    print("\nDataset de profiling salvo!")
    print(f"Arquivo: {output_path}")

    return df_profiling



# 6. Execução do Profiling
def main():

    print("\nIniciando Data Profiling...\n")

    # Momento de referência da execução
    processed_at = datetime.now(timezone.utc)

    datasets = load_data()

    all_metrics = []

    for dataset_name, df in datasets.items():

        inspect_structure(df, dataset_name)

        metrics = collect_profiling_metrics(
            df,
            dataset_name,
            processed_at
        )

        all_metrics.extend(metrics)

    df_profiling = save_profiling_dataset(
        all_metrics
    )

    print("\nResumo do Profiling:")
    print(df_profiling.head())

    print("\nData Profiling finalizado com sucesso!")



if __name__ == "__main__":
    main()
