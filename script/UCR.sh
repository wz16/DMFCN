
export CUDA_VISIBLE_DEVICES=0

declare -a datasets=("Adiac" "ArrowHead" "Beef" "BeetleFly" "BirdChicken" "Car" "CBF" "ChlorineConcentration" "CinCECGTorso" "Coffee" "Computers"  \
"CricketX" "CricketY" "CricketZ" "DiatomSizeReduction" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxOutlineCorrect" "DistalPhalanxTW" "Earthquakes" "ECG200"  \
"ECG5000" "ECGFiveDays" "ElectricDevices" "FaceAll" "FaceFour" "FacesUCR" "FiftyWords" "Fish" "FordA" "FordB" "GunPoint" "Ham" "HandOutlines" "Haptics"  \
"Herring" "InlineSkate" "InsectWingbeatSound" "ItalyPowerDemand" "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat"  \
"MedicalImages" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2"  \
"OliveOil" "OSULeaf" "PhalangesOutlinesCorrect" "Phoneme" "Plane" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxTW"  \
"RefrigerationDevices" "ScreenType" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SonyAIBORobotSurface1" "SonyAIBORobotSurface2" "StarLightCurves" "Strawberry"  \
"SwedishLeaf" "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace" "TwoLeadECG" "TwoPatterns" "UWaveGestureLibraryAll" "UWaveGestureLibraryX"  \
"UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine" "WordSynonyms" "Worms" "WormsTwoClass" "Yoga" "ACSF1" "AllGestureWiimoteX" "AllGestureWiimoteY"  \
"AllGestureWiimoteZ" "BME" "Chinatown" "Crop" "DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "EOGHorizontalSignal" "EOGVerticalSignal" "EthanolLevel"  \
"FreezerRegularTrain" "FreezerSmallTrain" "Fungi" "GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1" "GesturePebbleZ2" "GunPointAgeSpan"  \
"GunPointMaleVersusFemale" "GunPointOldVersusYoung" "HouseTwenty" "InsectEPGRegularTrain" "InsectEPGSmallTrain" "MelbournePedestrian" "MixedShapesRegularTrain"  \
"MixedShapesSmallTrain" "PickupGestureWiimoteZ" "PigAirwayPressure" "PigArtPressure" "PigCVP" "PLAID" "PowerCons" "Rock" "SemgHandGenderCh2" "SemgHandMovementCh2"  \
"SemgHandSubjectCh2" "ShakeGestureWiimoteZ" "SmoothSubspace" "UMD")

# uncomment the following line to run other models
# declare -a models=("DMFCN_1block" "DMFCN_2block" "DMFCN_3block" "DMFCN_4block" "DMFCN_3block_fourier" "DMFCN_3block_autocorre" "TimesNet" "OS_CNN" )

# standard DMFCN model with 3 blocks
declare -a models=("DMFCN_3block")

for model in "${models[@]}";do
for dataset in "${datasets[@]}";do
location="./dataset/UCRArchive_2018/"$dataset"/"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $location \
  --model_id $dataset \
  --model $model \
  --data UCR \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --log_name "result_ucr_"$model".txt"
done
done
