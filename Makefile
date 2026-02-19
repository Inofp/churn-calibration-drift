.PHONY: data train calibrate eval drift

data:
	python scripts/generate_data.py --out_dir data --seed 42

train:
	python scripts/train.py --data_dir data --out_dir artifacts

calibrate:
	python scripts/calibrate.py --data_dir data --in_dir artifacts --out_dir artifacts --method isotonic

eval:
	python scripts/evaluate.py --data_dir data --artifacts_dir artifacts --out_dir artifacts

drift:
	python scripts/drift_report.py --data_dir data --out_dir artifacts
