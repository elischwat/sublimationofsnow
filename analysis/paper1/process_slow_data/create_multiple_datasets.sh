#!/usr/bin/env bash

# Run 18 different configurations

# Choices are titled like this:
# despiking/filter-snowfall/filter-flags
# none/no/36000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall False \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_1_mm.csv" \
--snowfall-mask-str	'no' \
--percentage-diag 36000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/no/9000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall False \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_1_mm.csv" \
--snowfall-mask-str	'no' \
--percentage-diag 9000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/no/3600
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall False \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_1_mm.csv" \
--snowfall-mask-str	'no' \
--percentage-diag 3600 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/yes w23_precipitation_mask_0_mm/36000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_mm.csv" \
--snowfall-mask-str	'0mm' \
--percentage-diag 36000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/yes w23_precipitation_mask_0_mm/9000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_mm.csv" \
--snowfall-mask-str	'0mm' \
--percentage-diag 9000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/yes w23_precipitation_mask_0_mm/3600
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_mm.csv" \
--snowfall-mask-str	'0mm' \
--percentage-diag 3600 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/yes w23_precipitation_mask_0_5_mm/36000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'0.5mm' \
--percentage-diag 36000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/yes w23_precipitation_mask_0_5_mm/9000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'0.5mm' \
--percentage-diag 9000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# none/yes w23_precipitation_mask_0_5_mm/3600
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min" \
--filtering-str 'nodespiking' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'0.5mm' \
--percentage-diag 3600 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/no/36000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall False \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'no' \
--percentage-diag 36000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/no/9000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall False \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'no' \
--percentage-diag 9000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/no/3600
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall False \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'no' \
--percentage-diag 3600 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/yes w23_precipitation_mask_0_mm/36000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_mm.csv" \
--snowfall-mask-str	'0mm' \
--percentage-diag 36000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/yes w23_precipitation_mask_0_mm/9000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_mm.csv" \
--snowfall-mask-str	'0mm' \
--percentage-diag 9000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/yes w23_precipitation_mask_0_mm/3600
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_mm.csv" \
--snowfall-mask-str	'0mm' \
--percentage-diag 3600 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/yes w23_precipitation_mask_0_5_mm/36000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'0.5mm' \
--percentage-diag 36000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/yes w23_precipitation_mask_0_5_mm/9000
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'0.5mm' \
--percentage-diag 9000 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/" \
&&\
# q7/yes w23_precipitation_mask_0_5_mm/3600
python create_turbulence_dataset_30min_straightup.py \
--planar-fitted-dir "/Users/elischwat/Development/data/sublimationofsnow/planar_fit_processed_30min_despiked_q7" \
--filtering-str 'q7' \
--filter-snowfall True \
--snowfall-mask-file "/Users/elischwat/Development/data/sublimationofsnow/precipitation_masks/w23_precipitation_mask_0_5_mm.csv" \
--snowfall-mask-str	'0.5mm' \
--percentage-diag 3600 \
--output-dir "/Users/elischwat/Development/sublimationofsnow/analysis/paper1/turb_datasets/"