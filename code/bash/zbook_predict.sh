TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/validation"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/physicsBased/predictions/"
BATCH_SIZE=8
N_JOBS=2
BASE_CONFIG="/home/manolotis/sandbox/robustness_benchmark/physicsBased/code/configs/predict.yaml"
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/physicsBased/code/predict.py"

python $BASE_SCRIPT \
  --config $BASE_CONFIG \
  --test-data-path "$TEST_DATA_PATH" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS --out-path $OUT_PATH \
