# MNIST

### Training
##### 1. Setting up the Environment
Using `anaconda` as a distribution of `python 2.7` is recommended.

Running the file `setup_traning_env.sh` will install all the required dependencies to start the training process.

*NOTE: I have an commented code for installing pytorch on a CPU, make sure to uncomment it and remove the line above to run code on local machine.*

Additionally `requirements.txt` contains the list of all the dependencies but due to the complexity involved in some packages using the shell scripts is recommended.

##### 2. Hyperparameters and data preparation

###### Input: 
Images are 28x28 dimension with 60,000 samples for train and 10,000 for test.

`torchvision` is the library which is used to download the *MNIST* data.

###### Hyperparameters:
Here are  parameters that can be passed running the training script:

* `--batch`, `-b` : input batch size for training (default: 32)
* `--test_batch_size`, `-test_b` : input batch size for testing (default: 512)
* `--epochs`, `-e` : number of epochs to train (default: 10)
* `--learning_rate`, `-lr` : learning rate (default: 0.01)
* `--decay_gamma`, `-dg` : decay parameter to add this amount of decay (default:0.1)
* `--decay`, `-d` :how many epochs before the learning rate decays (default: 1000)
* `--optimizer`, `-opt` : Optimizer of choice (default: sgd)
* `--momentum`, `-m` : SGD momentum (default: 0.9)
* `--weight_decay`, `-wd` : SGD weight decay (default: 5e-4)
* `--log_dir`, `-log` : SGD weight decay (default: 5e-4)
* `--resume`, `-r` : Checkpoint file name. It ends with .ckpt (default: None)
* `--help`, `-h`: Get a help menu.

##### 3. Training the model

To kick off training: ``python train.py`` with any of the above paramters. 

##### 4. Evaluating the model performance

To evaluate the performance there are two tools that were built:

###### I. Progress bar to check the performance of the current training:
 
 This can be seen in the terminal after kicking off the model training.
 
###### II. Tensorboard to check the performance of the different training:
 
Using tensorboard:  `tensorboard --logdir runs/` in terminal and go to `http://localhost:6006`.

Currently the performance is measured by the `loss` and `accuracy` of train and `accuracy` of test.

There is a graph of the network also which is interactive to step through each block in the Network.

*NOTE: Using tensorboard all the different runs' accuracy can be compared.*


### Inferencing Demo:

##### 1. Setting up the environment
Running the file `setup_infer_env.sh` will install all the required dependencies.

Uses `onnx` and `Caffe2` to create prodcution models.

*NOTE: This approach is relatively new with pytorch and is being added as a module in the next release of pytorch 1.0*

##### 2. How to?

I have some sample data in the data folder that is not used for training. (./data/infer_data)

I also froze a trained model by:

                                    python create_prod_model.py -m {THE NAME OF THE MODEL}

Run inference of new images:

                                    python inference.py

*NOTE: Make sure the input images are of 28x28.*
