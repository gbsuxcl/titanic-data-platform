from pathlib import Path
from datetime import datetime, timezone
import subprocess
import zipfile
import json


# 1. Configuração dos caminhos
# Identifica a raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Diretório da camada BRONZE
BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "titanic"

# Caminho do arquivo ZIP
ZIP_PATH = BRONZE_PATH / "titanic.zip"

# Caminho do arquivo de metadados
METADATA_PATH = BRONZE_PATH / "ingestion_metadata.json"

# Arquivos esperados após a extração
EXPECTED_FILES = [
    "train.csv",
    "test.csv",
    "gender_submission.csv"
]



# 2. Download dos dados
def download_dataset():

    # Cria o diretório BRONZE caso ele não exista
    BRONZE_PATH.mkdir(parents=True, exist_ok=True)

    # Comando para baixar os dados do Kaggle
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "titanic",
        "-p",
        str(BRONZE_PATH)
    ]

    # Executa o comando
    subprocess.run(command, check=True)

    print("Download realizado com sucesso!")


# 3. Extração dos arquivos
def extract_dataset():

    # Verifica se o arquivo ZIP existe
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo ZIP não encontrado: {ZIP_PATH}"
        )

    # Extrai os arquivos
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(BRONZE_PATH)

    print("Arquivos extraídos com sucesso!")


# 4. Validação dos arquivos
def validate_dataset():

    # Verifica quais arquivos estão faltando
    missing_files = [
        file for file in EXPECTED_FILES
        if not (BRONZE_PATH / file).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Arquivos não encontrados: {missing_files}"
        )

    print("Validação dos arquivos concluída!")


# 5. Geração de metadados
def generate_metadata():

    metadata = {
        "source": "Kaggle",
        "dataset": "Titanic",
        "ingestion_date": datetime.now(timezone.utc).isoformat(),
        "files": EXPECTED_FILES,
        "status": "success"
    }

    # Salva os metadados em formato JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Metadados registrados com sucesso!")



# 6. Orquestração do pipeline
def main():

    print("\nIniciando pipeline de ingestão do Titanic...\n")

    # Etapa 1: Download
    download_dataset()

    # Etapa 2: Extração
    extract_dataset()

    # Etapa 3: Validação
    validate_dataset()

    # Etapa 4: Metadados
    generate_metadata()

    print("\nPipeline de ingestão finalizado com sucesso!")

if __name__ == "__main__":
    main()