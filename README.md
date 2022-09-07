# Endoscopy-single-image-depth-estimation-using-self-supervised-method
A Comparison Based Study On Depth Estimation of Monocular Endoscopic  Images using Self-supervised Learning Methods

Training of our implemented model from scratch

!python train.py --id_range 2 --input_downsampling 4.0 --network_downsampling 64 --adjacent_range 5 30 
           -      -input_size 256 320 --batch_size 2 --num_workers 2 --num_pre_workers 2 --validation_interval 1 
                 --display_interval 50  --dcl_weight 5.0 --sfl_weight 20.0 --max_lr 1.0e-3 --min_lr 1.0e-4 
                 --inlier_percentage 0.99 --visibility_overlap 30 --training_patient_id 1 --testing_patient_id 1 
                 --validation_patient_id 1 --number_epoch 51 --num_iter 100   --training_result_root "./Pre_trained_models" 
                 --training_data_root "./training_data_root"
                 
Training of our implemented model by loading pre-trained model//
!python train.py --id_range 2 --input_downsampling 4.0 --network_downsampling 64 --adjacent_range 5 30 
                 --input_size 256 320 --batch_size 2 --num_workers 2 --num_pre_workers 2 --validation_interval 1 
                 --display_interval 50  --dcl_weight 5.0 --sfl_weight 20.0 --max_lr 1.0e-3 --min_lr 1.0e-4 
                 --inlier_percentage 0.99 --visibility_overlap 30 --training_patient_id 1 --testing_patient_id 1 
                 --validation_patient_id 1 --number_epoch 51 --num_iter 100   --training_result_root "./Pre_trained_models" 
                 --training_data_root "./training_data_root" --load_trained_model
                 --trained_model_path "./Pre_trained_models/depth_estimation_train_run_8_3_18_8_test_id_[_1_]/"checkpoint_model_epoch_51_validation_0.08002564724948671.pt
                 
TensorBoard Visualisation of results//
!pip install tensorboardX
%load_ext tensorboard
%tensorboard --logdir './Pre_trained_models/depth_estimation_train_run_7_12_16_16_test_id_[_1_]'

Testing with pre-trained model with natural images//
!python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192

Refernce repositories:

Link for FC Densenet model genaration for training: 
https://github.com/SimJeg/FC-DenseNet   

Link for cyclical learning rate: 
https://github.com/bckenstler/CLR       

Link for single image depth estimation in endoscopy: 
https://github.com/lppllppl920/EndoscopyDepthEstimation-Pytorch  

Link for single image depth estimation in natural data set: 
https://github.com/nianticlabs/monodepth2
