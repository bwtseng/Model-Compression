# Model Compression

Currently, I focus on how to porting deep neural network to our local device with limited computation resource, this has been a hot and important research area since 2016, and it includes quantization, pruning light-weight model design. Actually, we should imitate the previous researcher for how they apply the traditional but powerful mechanisms to their local models before coming our own algorithm. That is totally why this repository creates.

# Directory Structure 

```
.Model Compression
├── __init__.py
├── checkpoint.py
├── config_logger.py
├── sensitivity.py
├── summary.py
├── summary_graph.py
├── tbbackend.py
├── thinning.py
├── utility.py
├── collector.py
├── compression_scheduler.py
├── netcollector.py
├── data_sampler.py
├── performance_tracker.py
├── policy.py
├── ptq_lapq.py
├── README.md
├── main.py
├── cifar10_download.py
├── model_zoo/
|	├── Imagenet
|	├── Cifar10
|	├── MNIST
|	├── VWW
|	└── __init__.py
├── quantization/
|	├── __init__.py
|	├── range_linear.py
|	└── ...
├── pruning/
|	├── __init__.py
|	├── automated_gradual_pruner.py
|	└── ...
├── modules/
|	├── __init__.py
|	├── gourping.py
|	└── ...
├── regularization/
|	├── __init__.py
|	├── regularizer.py
|	└── ...
├── yaml_file/
|	├── configle_file_example.yaml
|	└── ...
├── sensitivity_analysis/
|	├── summary csv/txt file
|	└── ...
└── ...

```

# Vision, Goal and Execution   

To enable our local models leverage the useful package Distiller without specified dataset and model listed in their repository, I study the distiller code whilst trying to incorporate into our local models with the easiest manner, that is, one can use the main.py to enjoy pruning, sensitivity analysis, and post-quantization so far . Note that it just supports  basic architecture and datasets, but it is more flexible to add new architecture, dataset and algorithm than distiller's repository.  More specifically,  by adding your new model script into the folder (model_zoo/data_folder), and type few lines to add the create_architecture_function in the python file (model_zoo/__init__.py). After completing these trivial steps, you are capable of using the command argument to train your specified architecture with the strategies configured into the YAML file created by each local user. 

```
python main.py -a resnet --dataset imagenet --data_dir path/to/your/data/folder -b 32 --compress path/to/your/cofig/yaml/file/ --out_dir
```

For executing, we can perform sensitive analysis (--sa), train (--train), test (--test) and post quantization (--pose_qunantization_test), please follow above command line. The rest of arguments are not listed here, you can just use --help to find out the detailed information. Note that you need to specify the absolute path to your model checkpoint to the argument --model_path for testing or continuing your training!

Our future goal may incorporate or even create more algorithms to the repository so that every user can enjoy the model compression in this kind of easy manner. As a consequence, we can put more emphasis on research rather than tool develop.

# Empirical Results

Currently, we experiment on ImageNet dataset using ShuffleNet, ResNet and SqueezeNet. Since it's time-consuming, so the result table may create in the near future.

Thank the [Github repo](https://github.com/huyvnphan/PyTorch-CIFAR10/tree/master/cifar10_models ) for giving us several pre-trained model and benchmark.

# Prerequisites

```
Plearse install the Distiller package following the official instruction.
```

# Acknowledgment

Very appreciate Intel distiller for the summarization and introduction on their page. If our revised version violates the license, please be kind to contact us and we will turn this repository into private mode immediately. Thank You!

Best regards,

BW Tseng