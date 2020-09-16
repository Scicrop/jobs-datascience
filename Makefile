make train_predict:
	python src/pipeline/train_test_split.py
	python src/pipeline/create_models.py 'data/train-validation/training-data.csv'
	python src/pipeline/predict.py 'data/Safra_2020.csv'

make docker:
	mkdir -p results
	docker build -t scicrop:pipline .
	docker run --gpus all -v ${CURDIR}/results:/src/results -it scicrop:pipline

make doc_files:
	mkdir -p ./doc/
	mkdir -p doc/notebooks
	mkdir -p doc/lib

	cp src/pipeline/lib/*.py doc/lib/
	rm doc/lib/__init__.py
	cp src/pipeline/lib/*.py doc/lib/
	cp src/notebooks/*.ipynb doc/notebooks
