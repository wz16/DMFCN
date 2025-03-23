export CUDA_VISIBLE_DEVICES=0




# declare -a datasets=()

# uncomment the following line to run other models
# declare -a models=("DMFCN_1block" "DMFCN_2block" "DMFCN_3block" "DMFCN_4block" "DMFCN_3block_fourier" "DMFCN_3block_autocorre" "TimesNet" "OS_CNN" )

# standard DMFCN model with 3 blocks
declare -a models=("DMFCN_3block")

declare -a datasets=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" \
"EthanolConcentration" "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
"NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" "InsectWingbeat" )

for dataset in "${datasets[@]}";do
for model in "${models[@]}";do
location="./dataset/UEAArchive_2018/""$dataset""/"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $location \
  --model_id $dataset \
  --model $model \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 20 \
  --log_name "result_uea_"$model".txt"
done  
done

# declare -a datasets=("InsectWingbeat")

# for dataset in "${datasets[@]}";do
# location="./dataset/UEAArchive_2018/""$dataset""/"
# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path $location \
#   --model_id $dataset \
#   --model LMFCN_3layer_fourier_nolayernormdropout \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 256 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 \
#   --log_name "result_classification_LMFCN_3layer_fourier_nolayernormdropout.txt"
# done