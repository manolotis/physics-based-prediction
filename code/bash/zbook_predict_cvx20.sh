TEST_DATA_PATH="/home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/data/prerendered/testing/"
OUT_PATH="/home/manolotis/sandbox/scenario_based_evaluation/physicsBased/predictions/"
BATCH_SIZE=8
N_JOBS=2
BASE_CONFIG="/home/manolotis/sandbox/scenario_based_evaluation/physicsBased/code/configs/predict_cvx20.yaml"
BASE_SCRIPT="/home/manolotis/sandbox/scenario_based_evaluation/physicsBased/code/predict.py"

python $BASE_SCRIPT \
  --config $BASE_CONFIG \
  --test-data-path "$TEST_DATA_PATH" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS --out-path $OUT_PATH \
