import os
import dask
from dask import dataframe as dd
from dask import delayed
from dask.distributed import Client, performance_report
from dask_jobqueue import SLURMCluster
import numpy as np
import pandas as pd
import glob
import psutil
import time
import tables_io
import getpass

# Identificar o path do usuário
user = getpass.getuser()
base_path = f'/scratch/users/{user}/random_from_files/'

# Criar pastas 'output' e 'logs' se não existirem
output_dir = os.path.join(base_path, 'output')
logs_dir = os.path.join(base_path, 'logs')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Configuração do SLURMCluster.
cluster = SLURMCluster(
    interface="ib0",    # Interface do Lustre
    queue='cpu_small',  # Substitua pelo nome da sua fila
    cores=56,           # Número de núcleos lógicos por nó
    processes=28,       # Número de processos por nó (um processo por núcleo)
    memory='100GB',     # Memória por nó
    walltime='01:00:00',  # Tempo máximo de execução
    job_extra_directives=[
        '--propagate',
        f'--output={output_dir}/dask_job_%j.out',  # Redireciona a saída para a pasta output
        f'--error={output_dir}/dask_job_%j.err'    # Redireciona o erro para a pasta output
    ],
)

# Escalando o cluster para usar X nós
cluster.scale(jobs=6)

# Definindo o client do Dask
client = Client(cluster)  

# Definição das variáveis pelo usuário
fraction = '<valor da fração de objetos desejada>'

# Caminho para o relatório de desempenho do Dask
performance_report_path = os.path.join(output_dir, f'performance_report_{fraction}.html')

with performance_report(filename=performance_report_path):
    # Obter lista de arquivos HDF5 na pasta
    file_list = glob.glob('/scratch/users/andreia.dourado/dados_tcc/skinny_table_tcc_cut/*.hdf5')
    
    # Ler todos os arquivos parquet com dask.
    #ddf = dd.read_parquet(file_list)
    def read_hdf5(file):
        x = tables_io.read(file)
        return pd.DataFrame(x)

    dfs = [delayed(read_hdf5)(file) for file in file_list]
    ddf = dd.from_delayed(dfs)
    
    # Fazer uma seleção aleatória.
    ddf_sampled = ddf.sample(frac=fraction, random_state=42)

    # Caminhos de saída
    output_path_parquet = f'/scratch/users/andreia.dourado/dados_tcc/random_sample_{fraction}.parquet'
    #output_path_csv =f'/scratch/users/andreia.dourado/dados_tcc/random_sample_{fraction}.csv'
    #output_path_hdf5 = #f'/scratch/users/andreia.dourado/dados_tcc/random_sample_{fraction}.hdf5'

    # Computando o dataframe.
    ddf_computed = ddf_sampled.compute()
    ddf_computed = ddf_computed.reset_index(drop=True)
    
    # Salvar os dados em Parquet
    ddf_computed.to_parquet(output_path_parquet, index=False)

    # Salvar os dados em CSV
    #ddf_computed.to_csv(output_path_csv, index=False)

    # Salvar os dados em HDF5
    #tables_io.write(ddf_computed, output_path_hdf5)

# Fechando o client
client.close()
cluster.close()
