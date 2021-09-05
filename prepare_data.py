import argparse
import logging
import pandas as pd
import numpy as np
from collections import Counter
import torch
import os

from tqdm.auto import tqdm

#Transformers
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate text embeddings based on Transformers."
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="File containing the input data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=250,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
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

    #Load training dataset
    logger.info("reading and splitting languages.")
    input_df = pd.read_csv(args.input_file)

    #split into languages
    #English
    input_en = input_df[input_df['Language'] == 'en'].copy()
    input_en = input_en.reset_index(drop=True)
    #take text after background (to clean a bit the abstract text, which contained a lot of junk sometimes)
    input_en['text'] = input_en['Abstract'].apply(lambda txt: txt[txt.lower().find('background') + 11:] if txt.lower().find('background') > -1 else txt)

    #Portuguese
    input_pt = input_df[input_df['Language'] == 'pt'].copy()
    input_pt = input_pt.reset_index(drop=True)
    #same as above, trying to clean a bit the abstract texts
    input_pt['text'] = input_pt['Abstract'].apply(lambda txt: txt[txt.lower().find('objetivo') + 9:] if txt.lower().find('objetivo') > -1 else txt)
    input_pt['text'] = input_pt['text'].apply(lambda txt: txt[txt.lower().find('introdução') + 11:] if txt.lower().find('introdução') > -1 else txt)

    #Spanish
    input_es = input_df[input_df['Language'] == 'es'].copy()
    input_es = input_es.reset_index(drop=True)
    #Spanish texts were the trickiest one, sometimes with descriptions in english and portuguese
    input_es['text'] = input_es['Abstract'].apply(lambda txt: txt[txt.lower().find('conclusão') + 10:] if txt.lower().find('conclusão') > -1 else txt)
    input_es['text'] = input_es['text'].apply(lambda txt: txt[txt.lower().find('conclusões') + 11:] if txt.lower().find('conclusões') > -1 else txt) 
    input_es['text'] = input_es['text'].apply(lambda txt: txt[:txt.lower().find('abstract')] if txt.lower().find('abstract') > txt.lower().find('objetivo') else txt)
    input_es['text'] = input_es['text'].apply(lambda txt: txt[txt.lower().find('objetivos') + 10:] if txt.lower().find('objetivos') > -1 else txt)
    input_es['text'] = input_es['text'].apply(lambda txt: txt[txt.lower().find('objetivo') + 9:] if txt.lower().find('objetivo') > -1 else txt)

    fl_type = 'train'
    if args.do_predict:
        fl_type = 'test'

    logger.info("Loading transformers models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check what device is available
    
    def emb_str_pooled(emb_model, tokenizer, input_str, max_length):
        
        # use a Transformers model along with a tokenizer to generate text embeddings
        
        emb_model.to(device)
        with torch.no_grad():
            input_ids = torch.tensor([tokenizer.encode(str(input_str), padding = 'max_length', max_length=max_length, truncation=True, add_special_tokens=False)], device=device).long()
            last_hidden_states = emb_model(input_ids)[1][0]
        return last_hidden_states.cpu().numpy()
    
    logger.info("English")
    model_name = 'distilroberta-base'
    en_tokenizer = AutoTokenizer.from_pretrained(model_name)
    en_emb_model = AutoModel.from_pretrained(model_name, output_attentions=False, output_hidden_states=False)
    
    input_embeddings_en = np.zeros([len(input_en), 768])
    for index, row in tqdm(input_en.iterrows(), total=len(input_en)):
        input_embeddings_en[index] = emb_str_pooled(emb_model=en_emb_model, tokenizer=en_tokenizer, input_str=row['text'], max_length=args.max_length)
        
    np.save(os.path.join(args.output_dir, f"{fl_type}_abstract_embedding_en.npy"), input_embeddings_en)
    
    logger.info("Portuguese")
    model_name = 'neuralmind/bert-base-portuguese-cased'
    pt_tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_emb_model = AutoModel.from_pretrained(model_name, output_attentions=False, output_hidden_states=False)
    
    input_embeddings_pt = np.zeros([len(input_pt), 768])
    for index, row in tqdm(input_pt.iterrows(), total=len(input_pt)):
        input_embeddings_pt[index] = emb_str_pooled(emb_model=pt_emb_model, tokenizer=pt_tokenizer, input_str=row['text'], max_length=args.max_length)
        
    np.save(os.path.join(args.output_dir, f"{fl_type}_abstract_embedding_pt.npy"), input_embeddings_pt)
    
    logger.info("Spanish")
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    es_tokenizer = AutoTokenizer.from_pretrained(model_name)
    es_emb_model = AutoModel.from_pretrained(model_name, output_attentions=False, output_hidden_states=False)
    input_embeddings_es = np.zeros([len(input_es), 768])
    for index, row in tqdm(input_es.iterrows(), total=len(input_es)):
        input_embeddings_es[index] = emb_str_pooled(emb_model=es_emb_model, tokenizer=es_tokenizer, input_str=row['text'], max_length=args.max_length)
        
    np.save(os.path.join(args.output_dir, f"{fl_type}_abstract_embedding_es.npy"), input_embeddings_es)
    
    logger.info("Embeddings done.")
    


if __name__ == "__main__":

    args = parse_args()

    main(args)