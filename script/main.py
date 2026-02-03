
from svoc.settings import get_settings, Settings
from svoc.utils import read_data_from_csv, get_logger
from svoc.orchestrator import svoc_knn, svoc_record_linkage

logger = get_logger()

def svoc_pipeline(
        settings: Settings
    ):

    df_input, df_benchmark = read_data_from_csv(settings)
    
    logger.info(
        "Input data loaded: %d input rows, %d benchmark rows",
        len(df_input),
        len(df_benchmark)
    )
    
    logger.info(
        "Computing postcode neighbourhoods using KNN (k=%d)",
        settings.N_NEIGHBORS
    )

    neighbors = svoc_knn(
        settings=settings, 
        df_input=df_input, 
        df_benchmark=df_benchmark, 
        k=settings.N_NEIGHBORS,
        save=False
    )

    logger.info("Postcode neighbourhoods computed (%d groups)", len(neighbors))

    logger.info("Starting record linkage")
    output = svoc_record_linkage(
        settings=settings, 
        df_input=df_input, 
        df_benchmark=df_benchmark,
        groups=neighbors,
        save=True
    )

    logger.info("Record linkage completed (%d matched rows)", len(output))
    logger.info("SVOC pipeline finished successfully")

def main():
    logger.info("Starting SVOC pipeline")
    settings = get_settings()
    logger.info("Settings loaded")
    svoc_pipeline(settings=settings)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("SVOC pipeline failed")
        raise
