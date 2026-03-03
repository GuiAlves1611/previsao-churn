import pandas as pd
from sqlalchemy import create_engine

#Infos 
user = "postgres"
password = "1234"
host = "localhost"
port = "5432"
database = "Churn"

#Conexão
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

#View Model
df = pd.read_sql("SELECT * FROM vw_telco_model", engine)

print(df.shape)

#Salvando

df.to_parquet("data/dataset_model.parquet", index=False)

print("Arquivo Salvo!")