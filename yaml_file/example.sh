python run_sensitivity_analysis.py $1 $2 $3 # Generate sensitivity table  
python run_greedy_selection_algo.py $1 $2 $3 # Generate sparsity level dict
python run_pruning.py $1 $2 $3 # Execute pruning algo.
python run_test.py $1 $2 $3 # Test pruned model.