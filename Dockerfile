FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src ./src
COPY data ./data
COPY results ./results

# command to run on container start
CMD	python src/pipeline/train_test_split.py && \
python3 src/pipeline/create_models.py 'data/train-validation/training-data.csv' && \
python3 src/pipeline/predict.py 'data/Safra_2020.csv'
