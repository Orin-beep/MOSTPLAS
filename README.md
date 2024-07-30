# MOSTPLAS
A self-correction multi-label learning model for plasmid host range prediction.

## Requirements
* Python >= 3.10  
* Pytorch >= 1.12.1  
* Biopython  

## Usage
* Download or clone this repository.
```Linux
git clone https://github.com/wzou96/MOSTPLAS
```  
* Prepare the fasta file of plasmid sequences and update the file in the folder ```input_fasta/```.
* Replace the path of the fasta file in the file ```test.py```.
```Python
input_path = './input_fasta/sequence.fasta'
```
* Run ```python test.py``` and obtain the prediction results in the folder ```output/prediction.txt```.

## Optional
* You can determine the threshold for making host range prediction in the file ```test.py```.
```Python
threshold = 0.4
```
* The default setting is 0.4.
