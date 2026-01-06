# Create the virtual environment
python -m venv venv

# Then activate it
.\venv\Scripts\activate

# Then install libery
pip install -r requirements.txt

# Run
python -m train.train_loop