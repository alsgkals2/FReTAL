<div align="center"># FReTAL: Generalizing Deepfake Detection using Knowledge Distillation and Representation Learning
 It is for transfer domain adaptation learning by minimizing the knowledge forgetting without any source data.

# Run
 For running the files, please input the commden.
 ## Unlearning

```
 'python run_FReTAL [number_gpu] [name_source] [name_target] [name_of_folder] [Bool (freezing some layer mode is 'True')] [subname_of_folder]'
 ```
 'number_gpu' --> gpu number\
 'name_source' --> prior dataset name\
 'name_target' --> current training dataset name\
 'name_of_folder' --> To distinguish the name of methods (KD, Finetuning, and FReTAL etc..)\
 'Bool' --> If you freezing some layer, 'True'. We set 'False' in the all experiments.\
 'subname_of_folder' --> 'name_of_folder' include this 'subname_of_folder'\
 \\
 Both make the folder ([name_source]_[name_target]) as well as load the dataset from the root directory path.
 The made folder is saved the weight of the student model while transfer learning.
[name of Foloder]

# README and description will be updated...
