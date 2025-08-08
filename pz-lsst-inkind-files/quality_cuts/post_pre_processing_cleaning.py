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

user = getpass.getuser()
base_path = f'/scratch/users/{user}/post_pre_processing/'
bands = ['u','g','r','i','z','y']
PATH_FOR_SKINNY='/scratch/users/andreia.dourado/dados_tcc/skinny_table_tcc/data/*.hdf5'
PATH_OUTPUT_DIR = f'/scratch/users/{user}/dados_tcc/skinny_table_tcc_cut'

def apply_validations(df):
    
    #Geral
    df = df[df['refExtendedness']==1]
    df = df[df['deblend_skipped']==False]
    df = df[df['i_cModel_flag']==False]
    df = df[df['i_centroid_flag']== False]
    df = df[df['i_blendedness'] < (10**(-0.375))]
    df = df[df['mag_i'] < 24.5]
    df = df[df['mag_i'] > 17]

    #S/N
    df['snr_g'] = 1 / (10 ** (0.4 * df['magerr_g']) - 1)
    df['snr_r'] = 1 / (10 ** (0.4 * df['magerr_r']) - 1)
    df['snr_z'] = 1 / (10 ** (0.4 * df['magerr_z']) - 1)
    df['snr_y'] = 1 / (10 ** (0.4 * df['magerr_y']) - 1)
    df['snr_i'] = 1 / (10 ** (0.4 * df['magerr_i']) - 1)

    snr_other_bands = (
    (df['snr_g'] > 5).astype(int) +
    (df['snr_r'] > 5).astype(int) +
    (df['snr_z'] > 5).astype(int) +
    (df['snr_y'] > 5).astype(int)
    )

    snr_i = df['snr_i'] > 10
    df['snr'] = (snr_i) & (snr_other_bands >= 2)
    df = df[df['snr'] == True]

    #Pixels
    df = df[df['i_pixelFlags_edge']==False]
    df = df[df['i_pixelFlags_clippedCenter']==False]
    df = df[df['i_pixelFlags_crCenter']==False]
    df = df[df['i_pixelFlags_interpolatedCenter']==False]
    df = df[df['i_pixelFlags_suspectCenter']==False]
    df = df[df['i_pixelFlags_saturatedCenter']==False]
    df = df[df['i_pixelFlags_offimage']== False]
    df = df[df['i_pixelFlags_bad']==False]
    df = df[df['i_pixelFlags_clipped']==False]
    df = df[df['i_pixelFlags_cr']==False]
    df = df[df['i_pixelFlags_interpolated']==False]
    df = df[df['i_pixelFlags_saturated']==False]
    df = df[df['i_pixelFlags_suspect']==False]
    df = df[df['i_centroid_flag']==False]

    
    
    df = df[['coord_dec', 'coord_ra','detect_isPrimary', 'mag_g', 'mag_i', 'mag_r', 'mag_u', 'mag_y', 'mag_z', 'magerr_g', 'magerr_i', 'magerr_r', 'magerr_u', 'magerr_y', 'magerr_z', 'objectId']]
    
    return df

def main():
    os.makedirs(base_path, exist_ok=True)
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
    
    # Caminho para o relatório de desempenho do Dask
    performance_report_path = os.path.join(output_dir, f'performance_report.html')
    
    with performance_report(filename=performance_report_path):
        file_list = glob.glob(PATH_FOR_SKINNY)
        files_output_dir = PATH_OUTPUT_DIR
        
        print(f"Quantidade arquivos entrada {len(file_list)}", flush=True)
        
        def read_and_filter_hdf5(file):
            df = tables_io.read(file, tables_io.types.PD_DATAFRAME)
            
            return apply_validations(df)
        
        def save_filtered_hdf5(file, df):
            output_path_hdf5 = os.path.join(files_output_dir, os.path.basename(file))
            tables_io.write(df, output_path_hdf5)
        
        
        def process_file(file):
            df = read_and_filter_hdf5(file)
            save_filtered_hdf5(file, df)
            return 1
    
        parts = [delayed(process_file)(file) for file in file_list]
    
        results = dask.compute(*parts)
        
        print(f"Finalizado, processado {sum(results)} arquivos")
    
    client.close()
    cluster.close()  
    
if __name__ == '__main__': main()
