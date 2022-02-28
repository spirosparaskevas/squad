# SQuADv2

## Installation

### Install PyEnv
Follow the instructions found [here](https://github.com/pyenv/pyenv).

### Install and activate python3.9
```bash
pyenv install 3.9.10
pyenv global 3.9.10
```

### Install poetry
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.2.0a2
```

### Install SQuAD
Inside the top level directory of the project run:
```bash
poetry install
```

### Activate virtualenv
```bash
poetry shell
```

## Data Preparation

### Download data
```bash
mkdir -p files
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -P files/
```

### Split data
```bash
squad create-datasets files/dev-v2.0.json
```

## Model commands

### Train model
```bash
squad train --learning-rate 0.00005 --batch-size 2 --accu-grads 32
```

### Predict answers for dev and test datasets
```bash
squad predict dev files/dev_predictions.json
squad predict test files/test_predictions.json
```

### Evaluate
```bash
python squad/evaluation.py files/dev_original_format.json files/dev_predictions.json
python squad/evaluation.py files/test_original_format.json files/test_predictions.json
```

## Exposure commands

### Run web server
```bash
uwsgi --http :8080 --wsgi-file squad/web.py --callable app
```

### Run streamlit client
```bash
streamlit run squad/streamlit_client.py
```