You can find the fully-implemented original U-net architecture under `orig_implementation_src`. It achieves a Dice score of 92% and an IoU score of 82%. 

The reason the final scores are not the same as in the actual paper is that we did not implement cross-validation to find the perfect train/test combination to increase model accuracy for benchmarks. You can see our full result scores/graphs under `orig_implementation_src/orig_impl_plots`.

### Instructions

To get the dataset, unzip the PhC-C2DH-U373.zip file in the data folder.

.tif files are the input image, and the label is man_seg000.tif

To pull the original model, you will need to run `git lfs install` and then `git lfs pull`, and it should appear in `orig_impl_checkpoints`.

To run inference of the initial U-net model, run `orig_implementation_src/inference.py`. An example of an inference run can be found in `orig_implementation_src/inference_example/inference_example.png`.

## Dependency installation
Install UV if not installed already: 'curl -LsSf https://astral.sh/uv/install.sh | sh'

Create and sync the environment: 'uv sync'

To run files inside the appropriate environment/folder: 
cd into the folder and run 'uv run python main.py'


## Training on Data Science Bowl 2018

1) Download the `stage1_train.zip` split from the Kaggle competition (https://www.kaggle.com/competitions/data-science-bowl-2018/data).  
2) Extract it to `data-science-bowl-2018/stage1_train` at the project root, so you have folders like `data-science-bowl-2018/stage1_train/<image_id>/{images,masks}`.  
3) In `src/config.py`, set `DATASET_CHOICE = "data-science-bowl-2018"` and adjust `DSB2018_TRAIN_ROOT` if you placed the data elsewhere.  
4) Run `python -m src.main` to start training on the Kaggle dataset using the existing U-Net pipeline, or `python -m src.train_2018` if you prefer the dedicated entrypoint.
