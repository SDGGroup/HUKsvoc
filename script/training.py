
from svoc.settings import get_settings
from svoc.utils import read_data
from svoc.supervised.enums import SupervisedModel
from svoc.supervised.match import train_all_models
from svoc.constants import DISTANCES

settings = get_settings()
# settings = get_settings("./config/dev2.yaml")

# Read Data
df_input, df_benchmark = read_data(settings)

# Train Models
models = train_all_models(
    df_input=df_input,
    input_cols_id_benchmark='sapcode',
    input_cols=settings.INPUT_COLUMNS_DICT,
    df_benchmark=df_benchmark,
    benchmark_cols=settings.BENCHMARK_COLUMNS_DICT,
    distances=DISTANCES,
    block_col=settings.BLOCK_COL,
    window=5,
    path_models=settings.SUPERVISED_MODELS_PATHS
)

models[SupervisedModel.LOGREG]
