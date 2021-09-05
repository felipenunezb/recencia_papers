# Recencia Papers

Below you can find a outline of how to reproduce my solution for the [Recencia Papers competition](https://www.datasource.ai/es/home/data-science-competitions-for-startups/predecir-el-puntaje-de-recencia-en-papers-de-investigacion).
If you run into any trouble with the setup/code or have any questions please contact me at f.nunezb@gmail.com

## ARCHIVE CONTENTS
* models.zip                  : trained model weights. Can also be downloaded from [GDrive](https://docs.google.com/uc?export=download&id=1CP_IMdiYYheBXq_cckBeSdnIrBgaXWaM)
* recencia_papers             : Github repo with codes and documentation

## HARDWARE/SOFTWARE: (The following specs were used to create the original solution)
* I used a common [Google Colab Notebook](https://colab.research.google.com) instance with GPU enabled (whichever Google provided at the time of run).

## MODEL BUILD:

1. Data PreProcessing (Generate Embeddings):

* function: prepare_data.py
* arguments:
    * input_file: folder where the train.csv has been saved
    * max_length: this controls how much text from the abstracts will be encoded. If shorter, the text will be padded. If longer, it will be truncated. Some models won't allow more than 512 tokens. Better results between 180 and 250.
    * output_dit: folder to store the results embeddings.
    * do_train/do_test: whether you are generating embeddings for the training of test set. This is important for the output files naming.

Training set
 ```python
!python /content/recencia_papers/prepare_data.py \
  --input_file '/content/train.csv' \
  --max_length 250 \
  --output_dir '/content/embeddings/' \
  --do_train
 ```

 Test set (notice the difference in input file and do_predict)
  ```python
!python /content/recencia_papers/prepare_data.py \
  --input_file '/content/test.csv' \
  --max_length 250 \
  --output_dir '/content/embeddings/' \
  --do_predict
 ```

2. Model training (Catboost):

*function: train.py
*arguments:
    * input_file: folder where the train.csv has been saved (usually same as above)
    * emb_folder: folder where the training text embeddings have been saved
    * output_dir: folder where to store the Catboost model weights
    * seed: defaults to 0, but could be other integers

```python
!python /content/recencia_papers/train.py \
  --input_file '/content/train.csv' \
  --emb_folder '/content/embeddings/' \
  --output_dir '/content/models/' \
  --seed 0
```

3. Generate predictions:

* function: predict.py
* arguments:
    * input_file: folder where the test.csv/final_test.csv files has been saved
    * emb_folder: folder where the test text embeddings have been saved
    * models_folder: folder where the Catboost model weights have been saved
    * output_dir: folder where to save the final predictions
    * do_predict

```python
!python /content/recencia_papers/predict.py \
  --input_file '/content/test.csv' \
  --emb_folder '/content/embeddings/' \
  --models_folder '/content/models/' \
  --output_dir '/content/predictions/'
```

## Example

There are 2 Notebook examples saved in the examples folder.

1. Recencia_Complete_demo.ipynb: Preprocess, train and predictions for the competition data. [Open in Google Colab](https://colab.research.google.com/drive/1KigW7wmRNurhIeh-QpQwX3awE6wkIAMG?usp=sharing)

2. Recencia_FinalTest.ipynb: Final test prediction. [Open in Google Colab](https://colab.research.google.com/drive/10alE-Og8zbNJMpUI828uXWy8J-xSu622?usp=sharing)

