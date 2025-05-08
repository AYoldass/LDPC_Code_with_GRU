# LDPC_Code_with_GRU

Basic Workflow

    Prepare data:
python Run_LDPC_GRU_System.py --mode prepare --data YN_output.xlsx

    Train model:
python Run_LDPC_GRU_System.py --mode train --data YN_output.xlsx --epochs 100

    Run inference:
python Run_LDPC_GRU_System.py --mode inference --data YN_output.xlsx

    Compare with AFF3CT:
python Run_LDPC_GRU_System.py --mode compare --aff3ct_results aff3ct_results.csv
