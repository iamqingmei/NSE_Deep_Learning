from app_dataset import save_app_pt_df
from manual_dataset import save_manual_pt_df
import logging
import sys


# create logger
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout)

window_size = 6
save_app_pt_df(window_size)
save_manual_pt_df(window_size)
