# FReTAL: Generalizing Deepfake Detection using Knowledge Distillation and Representation Learning
 It is for transfer domain adaptation learning by minimizing the knowledge forgetting without any source data.

# Run
 For running the files, please input the comment.
```
 'python run_FReTAL [number_gpu] [name_source] [name_target] [name_of_folder] [Bool (freezing some layer mode is 'True')] [subname_of_folder]'
 ```
**'number_gpu'** --> gpu number\
**'name_source'** --> prior dataset name\
**'name_target'** --> current training dataset name\
**'name_of_folder'**--> To distinguish the name of methods (KD, Finetuning, and FReTAL etc..)\
**'Bool'** --> If you freezing some layer, 'True'. We set 'False' in the all experiments.\
**'subname_of_folder'**--> 'name_of_folder' include this 'subname_of_folder'\
\\
It make the folder ([name_source]_[name_target]) as well as load the dataset from the root directory path.
Also, It is saved the weight of the student model while transfer learning.

# README and description will be updated...

## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@InProceedings{Kim_2021_CVPR,
    author    = {Kim, Minha and Tariq, Shahroz and Woo, Simon S.},
    title     = {FReTAL: Generalizing Deepfake Detection Using Knowledge Distillation and Representation Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {1001-1012}
}
```


