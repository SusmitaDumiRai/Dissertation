## Creating an intrusion detection system using supervised machine learning

#### Project overview
This project is a year-long dissertation submission for a BSc Computer Science in Swansea University that models an intrusion detection system using supervised machine learning. 

Three distinct models are used for the final comparison: random forest, support vector machine and an artifical neural network. Other models that were trained such as LSTMs were not used for final testing. 

This project also performed cyber attacks in an air-gapped, safe environment however the code (or tutorial) is not in this repository.

#### Dataset
The dataset used for this project is a collaboration between researchers that produced "a realistic cyber defense dataset". The dataset contains labelled statistical bi-directional flow data of multiple cyber attacks such as denial of service. The original PCAP files are also available.

The dataset: https://registry.opendata.aws/cse-cic-ids2018/

The research paper: https://www.scitepress.org/Papers/2018/66398/66398.pdf


### Code structure
The code is split into folders that it is used for.

It should be noted that the import path does not work after it was placed into these folders. Therefore, if you want to run the scripts you need to change the import paths of own files e.g. process_data.

#### Classifiers
All code here are related to creating a classifier which includes both classical machine learning algorithms such as random forest and neural networks.

classifier.py - this code is purely for random forest and support vector machine. It trains on the data on either basic split or in time series split.

Example code usage:
```   
python classifier.py -f dataset_location -o folder_to_save_data
```

ensemble.py - code for creating ensemble models. It is used by *nn_classifier.py*.

nn.py - code for creating neural network models. Supports creating an artificial neural network, using an pretrained model or creating a LSTM. It is used by *nn_classifier.py*.

nn_classifier.py - code for training neural networks. Trains using stratified cross validation.

Example code usage:
```
Train ANN: python nn_classifier.py -f dataset_location -o folder_to_save_data

Train singlular ensemble (uses logistic regression): python nn_classifier.py -f dataset_location -o folder_to_save_data -p pretrained_models_location -s

Train intergated ensemble:  python nn_classifier.py -f dataset_location -o folder_to_save_data -p pretrained_models_location -i
``` 

predict.py - code for predicting data using trained models

Example code usage:

```
Predict random forest: python predict.py -d dataset_location -m saved_rf_model_location -c

predict singular ensemble: python predict.py -d dataset_location -p location_of_pretrained_model  -g location_of_ensemble_model -s
```

#### Utility
All code here is used to handle the dataset. There are various different one-off scripts that were used when situation was required. Main script is *process_data.py*.

process_data.py - reads the dataset in for other Python scripts to be able to use the data. Also performs transformation on data if required such as normalising. 

#### Preliminary work
All code here belongs to preliminary work required to choose a dataset. Different datasets were compared using a subset of each.

#### GANN
The original aim of the dissertation was to use generative adversarial neural networks to test against false postives, however this was changed. All code here belongs to work completed before the aim changed.








