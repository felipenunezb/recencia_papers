import argparse
import logging
import pandas as pd
import numpy as np
from collections import Counter
import os

from tqdm.auto import tqdm

import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
    fl_type = 'train'

    #Load training dataset
    logger.info("Training")
    lang_suffix = ['en', 'pt', 'es']
    
    for lang in lang_suffix:
        
        logger.info(f"Training on: {lang} data.")
        
        input_sub = input_df[input_df['Language'] == lang].copy()
        input_sub = input_sub.reset_index(drop=True)
        
        train_embeddings = np.load(os.path.join(args.emb_folder, f"{fl_type}_abstract_embedding_{lang}.npy"))
        
        X = np.concatenate([np.expand_dims(np.array(input_sub['Year']), axis=1), train_embeddings], axis=1)
        Y = np.array(input_sub['total_rel_score'])
        
        ixs = np.arange(len(X))
        
        kf = KFold(n_splits=5)

        for fold_ix, (train_index, test_index) in tqdm(enumerate(kf.split(ixs))):
            X_train, y_train = X[train_index], Y[train_index]
            X_test, y_test = X[test_index], Y[test_index]

            #These hyperparameters were tuned/found previously
            if lang == 'en':
                model = cb.CatBoostRegressor(eval_metric='MSLE', grow_policy = "Lossguide",bootstrap_type='Poisson',random_seed=args.seed, task_type='GPU', od_type='Iter', od_wait=86, depth= 5, l2_leaf_reg=9.264310187012914, learning_rate= 0.025387195204195946, max_bin=299, model_size_reg=8.243800611558994, n_estimators=1000, num_leaves=39, subsample=0.7960154642805776)
            elif lang == 'pt':
                model = cb.CatBoostRegressor(eval_metric='MSLE', grow_policy = "Lossguide",bootstrap_type='Poisson',random_seed=args.seed, task_type='GPU', od_type='Iter', od_wait=86, depth=3, l2_leaf_reg=0.0932597179915462, learning_rate=0.07988103235176869, max_bin=152, model_size_reg=5.936090574644109, n_estimators=1000, num_leaves=17, subsample=0.8814987244793651)
            else:
                model = cb.CatBoostRegressor(eval_metric='MSLE', grow_policy = "Lossguide",bootstrap_type='Poisson', random_seed=args.seed, task_type='GPU', od_type='Iter', od_wait=86, depth=6, l2_leaf_reg=1.0550414450694667, learning_rate=0.021638099624990338, max_bin=299, model_size_reg=4.807836424649253, n_estimators=1000, num_leaves=38, subsample=0.6441769417293947)

            model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, early_stopping_rounds=86, verbose=False)
            model.save_model(os.path.join(args.output_dir, f"cat_{lang}_{fold_ix}.cbm"))

    logger.info("Training Done.")

if __name__ == "__main__":

    args = parse_args()

    main(args)