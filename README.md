# NSR_Captioning

This is the code for the paper of "Neural Symbolic Representation Learning for Image Captioning"


Data
The needed data is listed in the file of 'config.yml'.


1) input_rel_dir: the directory contains the '.npy' file of the relation probabilities.

      ---'.npy': the dimension is (n*n) x 51, n means the number of nodes in an image, 51 is the total relation classes(50 relation classes + 1 no relation)
      
      
2) input_json: it is a '.pkl' file and means the infomation of the dataset. The useful information in the file is listed as following.

      ---images: image list of the dataset, it is a list format, the length is 123287
      
         ---id: the id of the image, we load the node information of the image based on str(id) + '.pkl'
         
         ---split: the split of the image, 'test' or 'train' or 'val'
         
         ---file_path: the file path of the image '.jpg'
         
      ---ix_to_word: it is a dictionary format. the key is the index of word and the value is the word (string format). For example, 1:'chopped'. There are totally 9487 words in this paper.
      
      ---vocab_dic_glove: it is a matrix format(9488 x 300), 9487 word representaions + 1 all zeros
      
      
3) UP_box_dir: the directory contains '.pkl' files of node representations and node probabilities. the structre of one '.pkl' file contains

    ---'features': the representaions of the nodes, n x 2048 
    
    ---'boxes': the bboxes of the nodes, n x 4 
    
    ---'boxscores': the propabities of nodes on the node labels, n x 1601(1600 node labels + 1 no label)
    
    
4) input_label_h5: contains the information of the ground truth sentences for all images.

    ---'labels': n_seq x 16, n_seq is the number of sentences for all images, 16 is the number of words of one sentence.
    
    ---'label_start_ix': 123287 x 1, the start sentence index for one image
    
    ---'label_end_ix': 123278 x 1, the end sentence index for one image
    
    
5) Obj_glove_file: the word embeddings of the node labels (1601 x 300)

6) annFile: it is the file for evalation, contains ground truth sentences of evaluation images

7) VG_SGG_dictsFile: the index of relations 













