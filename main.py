import pandas as pd
import os, json, math, time, shutil
from pandas.io.parsers import TextFileReader
from pandas.core.series import Series
from pprint import pprint



def limpar_indicadores() -> None:
    
    last_value = None
    economic_indicators = "data/economic_indicators.csv"
    output_file = "clean_data/economic_indicators.csv"
    chunksize = 10000  # nu de linhas por vez
    indicators = pd.read_csv(economic_indicators, chunksize=chunksize)
    
    df_chunk: pd.DataFrame
    first = True
    
    for i, df_chunk in enumerate(indicators):   
        
        df_chunk.ffill(inplace=True)

        if last_value is not None:
            df_chunk.fillna(value=last_value, inplace=True)

        last_value = df_chunk.ffill().iloc[-1]

        if first:
            df_chunk.to_csv(output_file, index=False, mode='w')
            first = False
        else:
            df_chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        print(f"Chunk {i} processado:")
        print(df_chunk.head().to_string())
        

def save_stocks_dic_json(
        dataframe: TextFileReader
    ) -> dict:
    
    df_chunk: pd.DataFrame
    lista = []
    dic = {}
    for i, df_chunk in enumerate(dataframe):
        tmp = df_chunk['Symbol'].unique()

        for it in tmp:
            if it not in lista:
                lista.append(it)
                dic[it] = f"stocks/{it}.csv"

    # pprint(dic)
    # print(len(lista))
    with open("stock_path.json", "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)
    return dic


def get_stock_path_dic() -> dict:
    
    with open("stock_path.json", "r", encoding="utf-8") as f:
        dic = json.load(f)
    
    print(dic)
    return dic


def save_row_csv(
    row: Series,
    dic: dict
) -> None:
    # pprint(row)
    file = dic[row['Symbol']]
    first = not os.path.exists(file)  # True se arquivo n existe ainda
    
    row.to_frame().T.to_csv(
        file,
        mode="a",      # append se já existe
        header=first,  # escreve header na primeira vez
        index=False    # n cria indexes
    )


def separar_bovespa(
    dic: dict, 
    dataframe: TextFileReader,
    
) -> None:
    
    df_chunk: pd.DataFrame
    row: Series
    # stopper = 0
    for i, df_chunk in enumerate(dataframe):
        for idx, row in df_chunk.iterrows():
            save_row_csv(row, dic)
            # stopper+=1
            # if stopper == 20:
            # break
        # break


def limpar_bovespa(
    chunk_size: int   
) -> None:
    bovespa_stocks = "data/bovespa_stocks.csv"
    bovespa_stocks_dataframe = pd.read_csv(bovespa_stocks, chunksize=chunk_size)
    df_chunk: pd.DataFrame
    output_filename = 'clean_data/bovespa_stocks.csv'
    print(bovespa_stocks_dataframe)
    for i, df_chunk in enumerate(bovespa_stocks_dataframe):
        print(i)
        df_chunk["Date"] = pd.to_datetime(df_chunk["Date"], errors="coerce", utc=True) # Passa pra datetime utc
        df_chunk["Date"] = df_chunk["Date"].dt.tz_convert(None) # Remove fuso horario
        df_chunk["Date"] = df_chunk["Date"].dt.date # Remove as horas
        
        first = not os.path.exists(output_filename)
        df_chunk.to_csv(output_filename,mode='a', header=first, index=False)

def merge_dataframes(
    clean_indicators: str,
    dicicionarios_path: dict    
) -> None:
    
    chunksize = 10000
    
    indicators = pd.read_csv(clean_indicators, parse_dates=['Date'])
    
    for key, path in dicicionarios_path.items():
        
        output_filename = "stocks_indicators/"+path
        print(key, '->', path)
        stock = pd.read_csv(path, chunksize=chunksize, parse_dates=['Date'])
        
        for i, chunk in enumerate(stock):
            
            merged = pd.merge(
                chunk,
                indicators,
                on='Date',
                how='left'
            )
            
            first = not os.path.exists(output_filename)
            merged.to_csv(
                output_filename,
                mode='a',
                header=first,
                index=False
            )
            
def calc_correlation(quantidade, x, y, x_y, x_quad, y_quad) -> float:
    
    # print("Número:", quantidade, x, y, x_y, x_quad, y_quad)

    dividendo = (quantidade * x_y) - (x * y)
    a = (quantidade * x_quad) - (x * x)
    b = (quantidade * y_quad) - (y * y)

    if a <= 0 or b <= 0:
        return float('nan')

    divisor = math.sqrt(a * b)

    if divisor == 0:
        return float('nan')

    return dividendo / divisor


correlation_level = {
    -3: "STRONG NEGATIVE CORRELATION",
    -2: "MODERATE NEGATIVE CORRELATION",
    -1: "WEAK NEGATIVE CORRELATION",
    0: "NO CORRELATION",
    1: "WEAK POSITIVE CORRELATION",
    2: "MODERATE POSITIVE CORRELATION",
    3: "STRONG POSITIVE CORRELATION"
}

def get_corr_level(corr: float) -> int:
    
    if corr<=-0.7:
        return -3
    elif -0.7<corr<=-0.3:
        return -2
    elif -0.3<corr<0:
        return -1
    elif corr==0:
        return 0
    elif 0<corr<0.3:
        return 1
    elif 0.3<=corr<=0.7:
        return 2
    else:
        return 3
    
confiabiliade = {
    1: "POUCO CONFIAVEL",
    2: "MODERADAMENTE CONFIAVEL",
    3: "CONFIAVEL",
    4: "BASTANTE CONFIAVEL"
}

def get_correlation_confiability(corr_level, num_amostras) -> int:
    if(num_amostras<50):
        if corr_level == -3 or corr_level == 3:
            return 2
        else:
            return 1
    elif num_amostras<100:
        if corr_level == -3 or corr_level == 3:
            return 3
        elif corr_level == -2 or corr_level == 2:
            return 2
        else: 
            return 1
    elif num_amostras<200:
        if corr_level == -3 or corr_level == 3:
            return 4
        elif corr_level == -2 or corr_level == 2:
            return 3
        else: 
            return 2
        
    else:
        return 4

    
def merge_correlations(dic: dict) -> None:
    output_filename = "stocks_indicators_correlacionados.csv"
    first = True
    
    for key, path in dic.items():
        dataset = "stocks_indicators/correlation/" + key + ".csv"
        dataframe = pd.read_csv(dataset)

        # adiciona coluna "Empresa" com a key
        dataframe.insert(0, "Stock", key)

        dataframe.to_csv(
            output_filename,
            mode="a",
            header=first,
            index=False
        )
        first = False
        
        
def get_correlations(dic: dict) -> dict:
    merged_dataset = "stocks_indicators/"
    chunksize = 10000
    
    chunk: pd.DataFrame
    lista = {}
    for key, path in dic.items():
        
        quantidades = 0
        
        adj_close = adj_close_quad = 0
        close = close_quad = 0
        high = high_quad = 0
        low = low_quad = 0
        open_ = open_quad = 0
        volume = volume_quad = 0
        
        selic = selic_quad = 0
        ipca = ipca_quad = 0
        igp_m = igp_m_quad = 0
        inpc = inpc_quad = 0
        desemprego = desemprego_quad = 0
        
        selic_adj_close = selic_close = selic_high = selic_low = selic_open = selic_volume = 0
        ipca_adj_close = ipca_close = ipca_high = ipca_low = ipca_open = ipca_volume = 0
        igp_m_adj_close = igp_m_close = igp_m_high = igp_m_low = igp_m_open = igp_m_volume = 0
        inpc_adj_close = inpc_close = inpc_high = inpc_low = inpc_open = inpc_volume = 0
        desemprego_adj_close = desemprego_close = desemprego_high = desemprego_low = desemprego_open = desemprego_volume = 0


        merged_dataset = "stocks_indicators/"+path
        
        print(merged_dataset)
        merged_dataframe = pd.read_csv(merged_dataset, chunksize=chunksize)
        
        # Adj Close,Close,High,Low,Open,Volume,Taxa Selic,IPCA,IGP-M,INPC,Desemprego PNADC
        
        
        for i, chunk in enumerate(merged_dataframe):
            for idx, row in chunk.iterrows():
                
                # print(key, i, idx)
                if row.isna().any():
                    continue 
                quantidades+=1
                adj_close       += row['Adj Close']
                adj_close_quad  += row['Adj Close'] * row['Adj Close']
                close           += row['Close']
                close_quad      += row['Close'] * row['Close']
                high            += row['High']
                high_quad       += row['High'] * row['High']
                low             += row['Low']
                low_quad        += row['Low'] * row['Low']
                open_            += row['Open']
                open_quad       += row['Open'] * row['Open']
                volume          += row['Volume']
                volume_quad     += row['Volume'] * row['Volume']
                
                selic           += row['Taxa Selic']
                selic_quad      += row['Taxa Selic'] * row['Taxa Selic']
                ipca            += row['IPCA']
                ipca_quad       += row['IPCA'] * row['IPCA']
                igp_m           += row['IGP-M']
                igp_m_quad      += row['IGP-M'] * row['IGP-M']
                inpc            += row['INPC']
                inpc_quad       += row['INPC'] * row['INPC']
                desemprego      += row['Desemprego PNADC']
                desemprego_quad += row['Desemprego PNADC'] * row['Desemprego PNADC']
                
                selic_adj_close += row['Taxa Selic'] * row['Adj Close']
                selic_close     += row['Taxa Selic'] * row['Close']
                selic_high      += row['Taxa Selic'] * row['High']
                selic_low       += row['Taxa Selic'] * row['Low']
                selic_open      += row['Taxa Selic'] * row['Open']
                selic_volume    += row['Taxa Selic'] * row['Volume']
                
                ipca_adj_close += row['IPCA'] * row['Adj Close']
                ipca_close     += row['IPCA'] * row['Close']
                ipca_high      += row['IPCA'] * row['High']
                ipca_low       += row['IPCA'] * row['Low']
                ipca_open      += row['IPCA'] * row['Open']
                ipca_volume    += row['IPCA'] * row['Volume']
                
                igp_m_adj_close += row['IGP-M'] * row['Adj Close']
                igp_m_close     += row['IGP-M'] * row['Close']
                igp_m_high      += row['IGP-M'] * row['High']
                igp_m_low       += row['IGP-M'] * row['Low']
                igp_m_open      += row['IGP-M'] * row['Open']
                igp_m_volume    += row['IGP-M'] * row['Volume']
                
                inpc_adj_close += row['INPC'] * row['Adj Close']
                inpc_close     += row['INPC'] * row['Close']
                inpc_high      += row['INPC'] * row['High']
                inpc_low       += row['INPC'] * row['Low']
                inpc_open      += row['INPC'] * row['Open']
                inpc_volume    += row['INPC'] * row['Volume']
                
                desemprego_adj_close += row['Desemprego PNADC'] * row['Adj Close']
                desemprego_close     += row['Desemprego PNADC'] * row['Close']
                desemprego_high      += row['Desemprego PNADC'] * row['High']
                desemprego_low       += row['Desemprego PNADC'] * row['Low']
                desemprego_open      += row['Desemprego PNADC'] * row['Open']
                desemprego_volume    += row['Desemprego PNADC'] * row['Volume']
                
                
        n = quantidades

        corr_selic_adj_close = calc_correlation(n, selic, adj_close, selic_adj_close, selic_quad, adj_close_quad)
        corr_selic_adj_close_level = get_corr_level(corr_selic_adj_close)
        corr_selic_adj_close_confiability = get_correlation_confiability(corr_selic_adj_close_level, n)
        corr_selic_close     = calc_correlation(n, selic, close, selic_close, selic_quad, close_quad)
        corr_selic_close_level = get_corr_level(corr_selic_close)
        corr_selic_close_confiability = get_correlation_confiability(corr_selic_close_level, n)
        corr_selic_high      = calc_correlation(n, selic, high, selic_high, selic_quad, high_quad)
        corr_selic_high_level = get_corr_level(corr_selic_high)
        corr_selic_high_confiability = get_correlation_confiability(corr_selic_high_level, n)
        corr_selic_low       = calc_correlation(n, selic, low, selic_low, selic_quad, low_quad)
        corr_selic_low_level = get_corr_level(corr_selic_low)
        corr_selic_low_confiability = get_correlation_confiability(corr_selic_low_level, n)
        corr_selic_open      = calc_correlation(n, selic, open_, selic_open, selic_quad, open_quad)
        corr_selic_open_level = get_corr_level(corr_selic_open)
        corr_selic_open_confiability = get_correlation_confiability(corr_selic_open_level, n)
        corr_selic_volume    = calc_correlation(n, selic, volume, selic_volume, selic_quad, volume_quad)
        corr_selic_volume_level = get_corr_level(corr_selic_volume)
        corr_selic_volume_confiability = get_correlation_confiability(corr_selic_volume_level, n)

        corr_ipca_adj_close  = calc_correlation(n, ipca, adj_close, ipca_adj_close, ipca_quad, adj_close_quad)
        corr_ipca_adj_close_level = get_corr_level(corr_ipca_adj_close)
        corr_ipca_adj_close_confiability = get_correlation_confiability(corr_ipca_adj_close_level, n)
        corr_ipca_close      = calc_correlation(n, ipca, close, ipca_close, ipca_quad, close_quad)
        corr_ipca_close_level = get_corr_level(corr_ipca_close)
        corr_ipca_close_confiability = get_correlation_confiability(corr_ipca_close_level, n)
        corr_ipca_high       = calc_correlation(n, ipca, high, ipca_high, ipca_quad, high_quad)
        corr_ipca_high_level = get_corr_level(corr_ipca_high)
        corr_ipca_high_confiability = get_correlation_confiability(corr_ipca_high_level, n)
        corr_ipca_low        = calc_correlation(n, ipca, low, ipca_low, ipca_quad, low_quad)
        corr_ipca_low_level = get_corr_level(corr_ipca_low)
        corr_ipca_low_confiability = get_correlation_confiability(corr_ipca_low_level, n)
        corr_ipca_open       = calc_correlation(n, ipca, open_, ipca_open, ipca_quad, open_quad)
        corr_ipca_open_level = get_corr_level(corr_ipca_open)
        corr_ipca_open_confiability = get_correlation_confiability(corr_ipca_open_level, n)
        corr_ipca_volume     = calc_correlation(n, ipca, volume, ipca_volume, ipca_quad, volume_quad)
        corr_ipca_volume_level = get_corr_level(corr_ipca_volume)
        corr_ipca_volume_confiability = get_correlation_confiability(corr_ipca_volume_level, n)

        corr_igp_m_adj_close  = calc_correlation(n, igp_m, adj_close, igp_m_adj_close, igp_m_quad, adj_close_quad)
        corr_igp_m_adj_close_level = get_corr_level(corr_igp_m_adj_close)
        corr_igp_m_adj_close_confiability = get_correlation_confiability(corr_igp_m_adj_close_level, n)
        corr_igp_m_close      = calc_correlation(n, igp_m, close, igp_m_close, igp_m_quad, close_quad)
        corr_igp_m_close_level = get_corr_level(corr_igp_m_close)
        corr_igp_m_close_confiability = get_correlation_confiability(corr_igp_m_close_level, n)
        corr_igp_m_high       = calc_correlation(n, igp_m, high, igp_m_high, igp_m_quad, high_quad)
        corr_igp_m_high_level = get_corr_level(corr_igp_m_high)
        corr_igp_m_high_confiability = get_correlation_confiability(corr_igp_m_high_level, n)
        corr_igp_m_low        = calc_correlation(n, igp_m, low, igp_m_low, igp_m_quad, low_quad)
        corr_igp_m_low_level = get_corr_level(corr_igp_m_low)
        corr_igp_m_low_confiability = get_correlation_confiability(corr_igp_m_low_level, n)
        corr_igp_m_open       = calc_correlation(n, igp_m, open_, igp_m_open, igp_m_quad, open_quad)
        corr_igp_m_open_level = get_corr_level(corr_igp_m_open)
        corr_igp_m_open_confiability = get_correlation_confiability(corr_igp_m_open_level, n)
        corr_igp_m_volume     = calc_correlation(n, igp_m, volume, igp_m_volume, igp_m_quad, volume_quad)
        corr_igp_m_volume_level = get_corr_level(corr_igp_m_volume)
        corr_igp_m_volume_confiability = get_correlation_confiability(corr_igp_m_volume_level, n)

        corr_inpc_adj_close  = calc_correlation(n, inpc, adj_close, inpc_adj_close, inpc_quad, adj_close_quad)
        corr_inpc_adj_close_level = get_corr_level(corr_inpc_adj_close)
        corr_inpc_adj_close_confiability = get_correlation_confiability(corr_inpc_adj_close_level, n)
        corr_inpc_close      = calc_correlation(n, inpc, close, inpc_close, inpc_quad, close_quad)
        corr_inpc_close_level = get_corr_level(corr_inpc_close)
        corr_inpc_close_confiability = get_correlation_confiability(corr_inpc_close_level, n)
        corr_inpc_high       = calc_correlation(n, inpc, high, inpc_high, inpc_quad, high_quad)
        corr_inpc_high_level = get_corr_level(corr_inpc_high)
        corr_inpc_high_confiability = get_correlation_confiability(corr_inpc_high_level, n)
        corr_inpc_low        = calc_correlation(n, inpc, low, inpc_low, inpc_quad, low_quad)
        corr_inpc_low_level = get_corr_level(corr_inpc_low)
        corr_inpc_low_confiability = get_correlation_confiability(corr_inpc_low_level, n)
        corr_inpc_open       = calc_correlation(n, inpc, open_, inpc_open, inpc_quad, open_quad)
        corr_inpc_open_level = get_corr_level(corr_inpc_open)
        corr_inpc_open_confiability = get_correlation_confiability(corr_inpc_open_level, n)
        corr_inpc_volume     = calc_correlation(n, inpc, volume, inpc_volume, inpc_quad, volume_quad)
        corr_inpc_volume_level = get_corr_level(corr_inpc_volume)
        corr_inpc_volume_confiability = get_correlation_confiability(corr_inpc_volume_level, n)

        corr_desemp_adj_close = calc_correlation(n, desemprego, adj_close, desemprego_adj_close, desemprego_quad, adj_close_quad)
        corr_desemp_adj_close_level = get_corr_level(corr_desemp_adj_close)
        corr_desemp_adj_close_confiability = get_correlation_confiability(corr_desemp_adj_close_level, n)
        corr_desemp_close     = calc_correlation(n, desemprego, close, desemprego_close, desemprego_quad, close_quad)
        corr_desemp_close_level = get_corr_level(corr_desemp_close)
        corr_desemp_close_confiability = get_correlation_confiability(corr_desemp_close_level, n)
        corr_desemp_high      = calc_correlation(n, desemprego, high, desemprego_high, desemprego_quad, high_quad)
        corr_desemp_high_level = get_corr_level(corr_desemp_high)
        corr_desemp_high_confiability = get_correlation_confiability(corr_desemp_high_level, n)
        corr_desemp_low       = calc_correlation(n, desemprego, low, desemprego_low, desemprego_quad, low_quad)
        corr_desemp_low_level = get_corr_level(corr_desemp_low)
        corr_desemp_low_confiability = get_correlation_confiability(corr_desemp_low_level, n)
        corr_desemp_open      = calc_correlation(n, desemprego, open_, desemprego_open, desemprego_quad, open_quad)
        corr_desemp_open_level = get_corr_level(corr_desemp_open)
        corr_desemp_open_confiability = get_correlation_confiability(corr_desemp_open_level, n)
        corr_desemp_volume    = calc_correlation(n, desemprego, volume, desemprego_volume, desemprego_quad, volume_quad)
        corr_desemp_volume_level = get_corr_level(corr_desemp_volume)
        corr_desemp_volume_confiability = get_correlation_confiability(corr_desemp_volume_level, n)
        
        correlacoes = {
            
            "Selic vs Adj Close": [corr_selic_adj_close, corr_selic_adj_close_level,correlation_level[corr_selic_adj_close_level], n, corr_selic_adj_close_confiability, confiabiliade[corr_selic_adj_close_confiability]],
            "Selic vs Close": [corr_selic_close, corr_selic_close_level,correlation_level[corr_selic_close_level], n, corr_selic_close_confiability, confiabiliade[corr_selic_close_confiability]],
            "Selic vs High": [corr_selic_high, corr_selic_high_level, correlation_level[corr_selic_high_level], n, corr_selic_high_confiability, confiabiliade[corr_selic_high_confiability]],
            "Selic vs Low": [corr_selic_low, corr_selic_low_level, correlation_level[corr_selic_low_level], n, corr_selic_low_confiability, confiabiliade[corr_selic_low_confiability]],
            "Selic vs Open": [corr_selic_open, corr_selic_open_level, correlation_level[corr_selic_open_level], n, corr_selic_open_confiability, confiabiliade[corr_selic_open_confiability]],
            "Selic vs Volume": [corr_selic_volume, corr_selic_volume_level, correlation_level[corr_selic_volume_level], n, corr_selic_volume_confiability, confiabiliade[corr_selic_volume_confiability]],
            
            "IPCA vs Adj Close": [corr_ipca_adj_close, corr_ipca_adj_close_level, correlation_level[corr_ipca_adj_close_level], n, corr_ipca_adj_close_confiability, confiabiliade[corr_ipca_adj_close_confiability]],
            "IPCA vs Close": [corr_ipca_close, corr_ipca_close_level, correlation_level[corr_ipca_close_level], n, corr_ipca_close_confiability, confiabiliade[corr_ipca_close_confiability]],
            "IPCA vs High": [corr_ipca_high, corr_ipca_high_level, correlation_level[corr_ipca_high_level], n, corr_ipca_high_confiability, confiabiliade[corr_ipca_high_confiability]],
            "IPCA vs Low": [corr_ipca_low, corr_ipca_low_level, correlation_level[corr_ipca_low_level], n, corr_ipca_low_confiability, confiabiliade[corr_ipca_low_confiability]],
            "IPCA vs Open": [corr_ipca_open, corr_ipca_open_level, correlation_level[corr_ipca_open_level], n, corr_ipca_open_confiability, confiabiliade[corr_ipca_open_confiability]],
            "IPCA vs Volume": [corr_ipca_volume, corr_ipca_volume_level, correlation_level[corr_ipca_volume_level], n, corr_ipca_volume_confiability, confiabiliade[corr_ipca_volume_confiability]],
            
            "IGP-M vs Adj Close": [corr_igp_m_adj_close, corr_igp_m_adj_close_level, correlation_level[corr_igp_m_adj_close_level], n, corr_igp_m_adj_close_confiability, confiabiliade[corr_igp_m_adj_close_confiability]],
            "IGP-M vs Close": [corr_igp_m_close, corr_igp_m_close_level, correlation_level[corr_igp_m_close_level], n, corr_igp_m_close_confiability, confiabiliade[corr_igp_m_close_confiability]],
            "IGP-M vs High": [corr_igp_m_high, corr_igp_m_high_level, correlation_level[corr_igp_m_high_level], n, corr_igp_m_high_confiability, confiabiliade[corr_igp_m_high_confiability]],
            "IGP-M vs Low": [corr_igp_m_low, corr_igp_m_low_level, correlation_level[corr_igp_m_low_level], n, corr_igp_m_low_confiability, confiabiliade[corr_igp_m_low_confiability]],
            "IGP-M vs Open": [corr_igp_m_open, corr_igp_m_open_level, correlation_level[corr_igp_m_open_level], n, corr_igp_m_open_confiability, confiabiliade[corr_igp_m_open_confiability]],
            "IGP-M vs Volume": [corr_igp_m_volume, corr_igp_m_volume_level, correlation_level[corr_igp_m_volume_level], n, corr_igp_m_volume_confiability, confiabiliade[corr_igp_m_volume_confiability]],
            
            "INPC vs Adj Close": [corr_inpc_adj_close, corr_inpc_adj_close_level, correlation_level[corr_inpc_adj_close_level], n, corr_inpc_adj_close_confiability, confiabiliade[corr_inpc_adj_close_confiability]],
            "INPC vs Close": [corr_inpc_close, corr_inpc_close_level,correlation_level[corr_inpc_close_level], n, corr_inpc_close_confiability, confiabiliade[corr_inpc_close_confiability]],
            "INPC vs High": [corr_inpc_high, corr_inpc_high_level, correlation_level[corr_inpc_high_level], n, corr_inpc_high_confiability, confiabiliade[corr_inpc_high_confiability]],
            "INPC vs Low": [corr_inpc_low, corr_inpc_low_level, correlation_level[corr_inpc_low_level], n, corr_inpc_low_confiability, confiabiliade[corr_inpc_low_confiability]],
            "INPC vs Open": [corr_inpc_open, corr_inpc_open_level, correlation_level[corr_inpc_open_level], n, corr_inpc_open_confiability, confiabiliade[corr_inpc_open_confiability]],
            "INPC vs Volume": [corr_inpc_volume, corr_inpc_volume_level, correlation_level[corr_inpc_volume_level], n, corr_inpc_volume_confiability, confiabiliade[corr_inpc_volume_confiability]],
            
            "Desemprego PNADC vs Adj Close": [corr_desemp_adj_close, corr_desemp_adj_close_level, correlation_level[corr_desemp_adj_close_level], n ,corr_desemp_adj_close_confiability, confiabiliade[corr_desemp_adj_close_confiability]],
            "Desemprego PNADC vs Close": [corr_desemp_close, corr_desemp_close_level, correlation_level[corr_desemp_close_level], n, corr_desemp_close_confiability, confiabiliade[corr_desemp_close_confiability]],
            "Desemprego PNADC vs High": [corr_desemp_high, corr_desemp_high_level, correlation_level[corr_desemp_high_level], n, corr_desemp_high_confiability, confiabiliade[corr_desemp_high_confiability]],
            "Desemprego PNADC vs Low": [corr_desemp_low, corr_desemp_low_level, correlation_level[corr_desemp_low_level], n, corr_desemp_low_confiability, confiabiliade[corr_desemp_low_confiability]],
            "Desemprego PNADC vs Open": [corr_desemp_open, corr_desemp_open_level, correlation_level[corr_desemp_open_level], n, corr_desemp_open_confiability, confiabiliade[corr_desemp_open_confiability]],
            "Desemprego PNADC vs Volume": [corr_desemp_volume, corr_desemp_volume_level, correlation_level[corr_desemp_volume_level], n, corr_desemp_volume_confiability, confiabiliade[corr_desemp_volume_confiability]],
        }
        df = pd.DataFrame.from_dict(
            correlacoes, 
            orient="index",   # usa as chaves como índice
            columns=["Correlação", "Nível", "Nível str", "Amostras", "Confiabilidade", "Confiabilidade str"]
        )
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Comparação"}, inplace=True)
        
        df.to_csv(f"stocks_indicators/correlation/{key}.csv", index=False, encoding="utf-8")
        pprint(correlacoes)
        lista[key] = correlacoes
        # pprint(lista)
    # pprint(lista)
    
    with open("correlation_stock_indicators.json", "w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False, indent=4)
    return lista



# bovespa_stocks = "data/bovespa_stocks.csv"

# # df = pd.read_csv("data/bovespa_stocks.csv", nrows=10)

# # stocks = pd.read_csv(bovespa_stocks, parse_dates=['Date'])

def limpar_diretorio(path: str) -> None:

    if not os.path.isdir(path):
        raise ValueError(f"O caminho '{path}' n é um dir valido.")

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def main() -> None:
    
    
    start = time.time()
    chunk_size = 100000
    
    bovespa_stocks = "data/bovespa_stocks.csv"
    bovespa_stocks_dataframe = pd.read_csv(bovespa_stocks, chunksize=chunk_size) # Por algum motivo n consigo passar por funcao???
    
    save_stocks_dic_json(bovespa_stocks_dataframe)
    dicionario = get_stock_path_dic()
    print(1)
    limpar_bovespa(chunk_size)
    print(2)
    bovespa_stocks_clean = "clean_data/bovespa_stocks.csv"
    bovespa_stocks_clean_dataframe = pd.read_csv(bovespa_stocks_clean, chunksize=chunk_size)
    print(3)
    separar_bovespa(dicionario, bovespa_stocks_clean_dataframe)
    limpar_indicadores()
    print(4)
    economic_indicators_stocks_clean = "clean_data/economic_indicators.csv"
    print(5)
    merge_dataframes(economic_indicators_stocks_clean, dicionario)
    print(6)
    get_correlations(dicionario)
    print(7)
    merge_correlations(dicionario)
    print(8)
    end = time.time()
    
    limpar_diretorio("clean_data")
    limpar_diretorio("stocks")
    limpar_diretorio("stocks_indicators/correlation")
    limpar_diretorio("stocks_indicators/stocks")
    
    
    print(f"Tempo de execucao: {end-start:.4f} sec")



if __name__ == "__main__":
    main()
