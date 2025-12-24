docker build -t ml-experiment .
docker run -it --rm -v "$(pwd):/app" ml-experiment
python rf_tuned.py
python getmethod.py
python ml_tuned.py