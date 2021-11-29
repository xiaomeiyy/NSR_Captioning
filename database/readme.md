
1) input_rel_dir: the directory contains the '.npy' file of the relation probabilities.

      ---'.npy': the dimension is (n*n) x 51, n means the number of nodes in an image, 51 is the total relation classes(50 relation classes + 1 no relation)
      
      
2) [cocotalk_glove.pkl](https://drive.google.com/file/d/1hSLLO-ZP6qNpFOdJgIdMo7P-7JgR5Rzx/view?usp=sharing) means the infomation of the dataset.
      
      
3) UP_box_dir: the directory contains '.pkl' files of node representations and node probabilities. the structre of one '.pkl' file contains

    ---'features': the representaions of the nodes 
    
    ---'boxes': the bboxes of the nodes
    
    ---'boxscores': the propabities of nodes on the node labels
    
    
4) [cocotalk_label.h5](https://drive.google.com/file/d/1g6XYiCF7KTZS4HIQkiM2AtLT00lsYVC9/view?usp=sharing) contains the information of the ground truth sentences for all images.

    ---'labels': the ground truth labels;
    
    ---'label_start_ix': the start sentence index for one image;
    
    ---'label_end_ix': the end sentence index for one image.
    
    
5) objects1600_glove.pkl: the word embeddings of the node labels.

6) annFile: it is the file for evalation, contains ground truth sentences of evaluation images.

7) VG-SGG-dicts.json: the index of relations.



