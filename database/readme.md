
1) input_rel_dir: the directory contains the '.npy' file of the relation probabilities.

      ---'.npy': the dimension is (n*n) x 51, n means the number of nodes in an image, 51 is the total relation classes(50 relation classes + 1 no relation)
      
      
2) input_json: [cocotalk_glove.pkl](https://drive.google.com/file/d/1hSLLO-ZP6qNpFOdJgIdMo7P-7JgR5Rzx/view?usp=sharing) means the infomation of the dataset.
      
      
3) UP_box_dir: the directory contains '.pkl' files of node representations and node probabilities. the structre of one '.pkl' file contains

    ---'features': the representaions of the nodes, n x 2048 
    
    ---'boxes': the bboxes of the nodes, n x 4 
    
    ---'boxscores': the propabities of nodes on the node labels, n x 1601(1600 node labels + 1 no label)
    
    
4) input_label_h5: [cocotalk_label.h5](https://drive.google.com/file/d/1g6XYiCF7KTZS4HIQkiM2AtLT00lsYVC9/view?usp=sharing) contains the information of the ground truth sentences for all images.

    ---'labels': the ground truth labels;
    
    ---'label_start_ix': the start sentence index for one image;
    
    ---'label_end_ix': the end sentence index for one image.
    
    
5) Obj_glove_file: the word embeddings of the node labels.

6) annFile: it is the file for evalation, contains ground truth sentences of evaluation images.

7) VG_SGG_dictsFile: the index of relations.



