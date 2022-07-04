# Predicting whether a DNA sequence belongs to the SARS-CoV-2 (Covid-19)

## Introduction
The goal of the data challenge is to learn how to implement machine learning algorithms, understand them, and adapt them to structural data.
For this reason, we were given a sequence classification task: to predict whether a DNA sequence (or read) belongs to SARS-CoV-2 (Covid-19).

## Data description
The training and evaluation data are sets of DNA sequencing reads: short DNA fragments (~100-300 bp long), which come from sequencing experiments, or have been simulated from complete genomes. Some of these fragments come from Covid-19 genomes, others from humans or random bacteria.
The objective is to discriminate Covid-19 fragments, hence a binary classification task: the labels are either 1 if the fragment is identified as Covid-19, or 0 otherwise.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rrDMZm90kBRAh1HnLpI_5XhnVaTaIDxX?usp=sharing)

# Install the Project

```
$ git@github.com:Junior-081/SARS-CoV-2-Covid-19-DNA-sequence-prediction-with-Kernel-SVM.git
```

```
$ cd SARS-CoV-2-Covid-19-DNA-sequence-prediction-with-Kernel-SVM
```
# Virtual environment

## Mac OS

### Create virtual environment 

```
$ python3 -m venv NAME_ENV
```
### Activate your environment 

```
$ source NAME_ENV/bin/activate
```

## Linux OS

### Create virtual environment

```
$ conda create -n venv NAME_ENV
```

### Activate your environment 

```
$ activate NAME_ENV
```

# Requirement installations
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```
# Running the model
You make sure that you have version 3 of python

To train and Test
```
$ python3 main.py

```




# Author #
<div style="display:flex;align-items:center">

 <div style="display:flex;align-items:center">
  
 <div>
    <h5> <a href='https://github.com/Junior-081'> Junior Kaningini </a> </h5> <img src="junior.png" height= 7% width= 7%>
            
 </div>


</div>

</div>

