import argparse
import logging
import pandas as pd
import numpy as np
from collections import Counter
import os
from datetime import datetime

from tqdm.auto import tqdm

import catboost as cb

logger = logging.getLogger(__name__)

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train models."
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="File containing the input data."
    )
    parser.add_argument(
        "--emb_folder", type=str, default=None, help="Folder containing the training embedding data."
    )
    parser.add_argument(
        "--models_folder", type=str, default=None, help="Folder containing the trained models."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final embeddings.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Train or no",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Eval or no",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    input_df = pd.read_csv(args.input_file)
    fl_type = 'test'

    #Load training dataset
    logger.info("Training")
    lang_suffix = ['en', 'pt', 'es']
    
    ds_dict = {}
    preds_dict = {}
    
    for lang in lang_suffix:
        
        logger.info(f"Predicting on: {lang} data.")
        
        input_sub = input_df[input_df['Language'] == lang].copy()
        input_sub = input_sub.reset_index(drop=True)
        
        ds_dict[lang] = input_sub #save in dict, for future reference
        
        #Load embedding
        test_embeddings = np.load(os.path.join(args.output_dir, f"{fl_type}_abstract_embedding_{lang}.npy"))
        
        pred_folds = []
        model_inputs = np.concatenate([np.expand_dims(np.array(input_sub['Year']), axis=1), test_embeddings], axis=1)
        
        for fold in range(5):
            logger.info(f"Lang: {lang} - Fold: {fold}")
            model = cb.CatBoostRegressor()
            path_str = os.path.join(args.emb_folder, f"cat_{lang}_{fold}.cbm")
            model.load_model(path_str)

            prediction = model.predict(model_inputs, task_type='CPU')
            
            pred_folds.append(prediction)
        
        logger.info(f"Lang {lang}: predict the mean.")
        preds_dict[lang] = np.mean(np.array(pred_folds), axis=0)
        
    logger.info("Merge predictions into single file.")
    
    merged_preds = np.zeros(len(input_df))
    for lang in lang_suffix:
        
        for n, pred in enumerate(preds_dict[lang]):
            ix = ds_dict[lang].iloc[n]['id'] - 1
            merged_preds[ix] = pred
    
    final_preds = pd.DataFrame({'id': range(1, len(input_df)+1), 'total_rel_score': merged_preds})
    
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    final_preds.to_csv(os.path.join(args.output_dir, f"SampleSubmission_{date}.csv"), index=False)

    logger.info("Prediction Done.")

if __name__ == "__main__":

    args = parse_args()

    main(args)