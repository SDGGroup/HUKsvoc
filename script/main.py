from svoc.settings import get_settings
from svoc.utils import read_data_from_csv
from svoc.datapreparation import prepare_data
from svoc.rl import get_matches_with_clusters, prepare_output
from svoc.constants import DISTANCES, FILTERS_AUTO

def main():
    settings = get_settings()

    df_input, df_benchmark = read_data_from_csv(settings)

    df_benchmark_clean = prepare_data(
        df=df_benchmark, 
        dict_cols=settings.BENCHMARK_COLUMNS_DICT,
        )

    df_input_clean = prepare_data(
        df=df_input, 
        dict_cols=settings.INPUT_COLUMNS_DICT,
        )

    all_matches, features, remaining_features = get_matches_with_clusters(
        df_input=df_input_clean, 
        df_benchmark=df_benchmark_clean, 
        block_col=settings.BLOCK_COL, 
        distances=DISTANCES, 
        filters=FILTERS_AUTO,
        n_matches=settings.N_MATCHES, verbose=False,
        models_path_dict=settings.SUPERVISED_MODELS_PATHS
        )

    output = prepare_output(
        matches=all_matches,
        distances=DISTANCES,
        filters=FILTERS_AUTO
    )

    output.to_csv(settings.DATA_DIR / 'output.csv', index=False)    

if __name__ == "__main__":
    main()
