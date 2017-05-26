from data_retrieval.app_dataset import save_app_pt_df
from data_retrieval.manual_dataset import save_manual_pt_df
from data_retrieval.google_dataset import save_google_pt_df
import logging
import sys
import params


# create logger
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.WARNING, stream=sys.stdout)

window_size = params.window_size
# 18 minutes needed to get app and manual pt_df
save_app_pt_df(window_size)
logging.warning("Finished processing app point data frame")
save_manual_pt_df(window_size)
# 4 hours 30 minutes needed to get google pt_df
save_google_pt_df(window_size)
