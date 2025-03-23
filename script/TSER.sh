

export CUDA_VISIBLE_DEVICES=0

# uncomment the following line to run other models
# declare -a models=("DMFCN_1block" "DMFCN_2block" "DMFCN_3block" "DMFCN_4block" "DMFCN_3block_fourier" "DMFCN_3block_autocorre" "TimesNet" "OS_CNN" )

# standard DMFCN model with 3 blocks
declare -a models=("DMFCN_3block")


declare -a datasets=("AppliancesEnergy" "LiveFuelMoistureContent" "BeijingPM10Quality" "BeijingPM25Quality"  "IEEEPPG"  "BenzeneConcentration" "Covid3Month" "FloodModeling1" \
"FloodModeling2" "FloodModeling3" "HouseholdPowerConsumption1" "HouseholdPowerConsumption2")

for dataset in "${datasets[@]}";do
for model in "${models[@]}";do
location="./dataset/TSER/"$dataset"/"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --features MS \
  --task_name regression \
  --is_training 1 \
  --root_path $location \
  --model_id $dataset \
  --model $model \
  --data TSER \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 10 \
  --log_name "result_regression_"$model".txt"
done
done


declare -a datasets=("BIDMC32SpO2" "BIDMC32RR" "BIDMC32HR" "AustraliaRainfall" "NewsHeadlineSentiment" "NewsTitleSentiment" "PPGDalia")

for dataset in "${datasets[@]}";do
for model in "${models[@]}";do

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --features MS \
  --task_name regression \
  --is_training 1 \
  --root_path $location \
  --model_id $dataset \
  --model $model \
  --data TSER \
  --e_layers 3 \
  --batch_size 128 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 10 \
  --log_name "result_regression_"$model".txt"
done
done
