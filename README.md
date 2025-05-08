# LDPC_Code_with_GRU

Installation

    Install dependencies:


pip install -r requirements.txt

    (Optional) Install AFF3CT for comparison:


git clone https://github.com/aff3ct/aff3ct.git
cd aff3ct
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE='DEBUG'
make -j16

Usage
Basic Workflow

    Prepare data:


python Run_LDPC_GRU_System.py --mode prepare --data YN_output.xlsx

    Train model:


python Run_LDPC_GRU_System.py --mode train --data YN_output.xlsx --epochs 100

    Run inference:


python Run_LDPC_GRU_System.py --mode inference --data YN_output.xlsx

    Compare with AFF3CT:

python Run_LDPC_GRU_System.py --mode compare --aff3ct_results aff3ct_results.csv

Command Line Options
Option	Description	Default
--mode	Operation mode (prepare, train, test, inference, compare, all)	all
--data	Path to dataset file	YN_output.xlsx
--model	Path to model file	ldpc_gru_model.pth
--epochs	Number of training epochs	100
--aff3ct_results	Path to AFF3CT results file	aff3ct_results.csv
Data Structure

The input data should be in Excel format with following columns:
Column	Description
feature_0 to feature_N	LLR (Log-Likelihood Ratio) values
SNR	Signal-to-Noise Ratio (dB)
label	Error indicator (1: error, 0: no error)
Model Architecture
python

GRUModel(
  (gru): GRU(input_size, hidden_size=128, num_layers=2, batch_first=True)
  (fc): Linear(in_features=128, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)

Results

Example performance comparison:

BER Comparison
FER Comparison
Contributing

Contributions are welcome! Please follow these steps:

    Fork the repository

    Create a new branch (git checkout -b feature-branch)

    Commit your changes (git commit -am 'Add new feature')

    Push to the branch (git push origin feature-branch)

    Create a new Pull Request
