TEST_DATA_PATH="/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/testing"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/physicsBased/predictions/"
BATCH_SIZE=128
N_JOBS=48
BASE_CONFIG="/home/manolotis/sandbox/robustness_benchmark/physicsBased/code/configs/predict.yaml"
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/physicsBased/code/predict.py"

python $BASE_SCRIPT \
  --config $BASE_CONFIG \
  --test-data-path "$TEST_DATA_PATH" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS --out-path $OUT_PATH \
