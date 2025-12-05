"""
Entry point to train the U-Net on the Data Science Bowl 2018 dataset.
Ensure `data-science-bowl-2018/stage1_train` exists (see README for layout).
"""

from config import DATASET_CHOICE
from main import run_dsb_training


def main():
    # Allow reuse of the config toggle while making this a dedicated entrypoint.
    if DATASET_CHOICE != "data-science-bowl-2018":
        print(
            "Warning: DATASET_CHOICE is not set to 'data-science-bowl-2018'. "
            "Proceeding to run the DSB2018 pipeline."
        )
    run_dsb_training()


if __name__ == "__main__":
    main()
