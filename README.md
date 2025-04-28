# JTCSE
Code and CKPTs of paper JTCSE: Joint Tensor Modulus Constraints and Cross Attention for Unsupervised Contrastive Learning of Sentence Embeddings

# Introduction

This repository is belong to the paper titled "JTCSE: Joint Tensor Modulus Constraints and Cross Attention for Unsupervised Contrastive Learning of Sentence Embeddings".

JTCSE is a BERT-like model for computing sentence embedding vectors, trained using unsupervised contrastive learning.

JTCSE is an extension of the AAAI25 **Oral** paper [TNCSE](https://ojs.aaai.org/index.php/AAAI/article/view/34816).
# How to Use

We recommend you train JTCSE with RAM >= 48GB and GPU memory >= 24GB.

## Installation

You also need to make sure your python >= 3.6 and install py repositories in requirements.txt :
```bash
pip install -r requirements.txt
```

After installation, make sure you download models' [checkpoints and training datasets](https://huggingface.co/UCASzty) from HF and copy all the folders into the directory where the project resides. All the checkpoints you need are in these folders.
**The JTCSE training dataset is consistent with TNCSE.**
Please download the [SentEval](https://github.com/princeton-nlp/SimCSE/tree/main/SentEval) folder to the path.

## Direct Evaluation

#### We report the results directly below the command.

### Eval JTCSE_BERT
```bash
python evaluation_CKPT.py --model_name_or_path_1 JTCSE_saved_ckpt/JTCSE_BERT/JTCSE_bert_ckpt_encoder_1/ --model_name_or_path_2 JTCSE_saved_ckpt/JTCSE_BERT/JTCSE_bert_ckpt_encoder_2/ --tokenizer_path [your BERT-base-uncased path]
```
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 74.96 | 84.20 | 77.79  | 84.75 | 80.43 | 81.87        | 73.88           | 79.70 |

### Eval JTCSE_RoBERTa
```bash
python evaluation_CKPT.py --model_name_or_path_1 JTCSE_saved_ckpt/JTCSE_RoBERTa/JTCSE_roberta_ckpt_encoder_1/ --model_name_or_path_2 JTCSE_saved_ckpt/JTCSE_RoBERTa/JTCSE_roberta_ckpt_encoder_2/ --tokenizer_path [your RoBERTa-base path]
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 74.92 | 84.79 | 76.99 | 84.91 | 81.58 | 82.71        | 73.74           | 79.95 |

### Eval JTCSE_BERT_D
```bash
python evaluation_D.py --model_name_or_path JTCSE_D/BERT/JTCSE_BERT_D/
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 75.01 | 84.86 | 77.76 | 84.62 | 80.38 | 82.05        | 74.53           | 79.89 |

### Eval TNCSE_RoBERTa_D
```bash
python evaluation_D.py --model_name_or_path JTCSE_D/RoBERTa/JTCSE_RoBERTa_D/
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 75.42 | 85.36 | 77.31 | 85.04 | 81.72 | 82.91        | 74.46           | 80.32 |

### Eval JTCSE_BERT_UC
```bash
python ensemble_UC.py --model_type BERT --model_path1 JTCSE_saved_ckpt/JTCSE_BERT/JTCSE_bert_ckpt_encoder_1/ --model_path2 JTCSE_saved_ckpt/JTCSE_BERT/JTCSE_bert_ckpt_encoder_2/ --ensemble_UC [your InfoCSE-BERT path]
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 75.44 | 85.34 | 78.75  | 85.93 | 82.00 | 83.21         | 73.52           | 80.60 |

### Eval JTCSE_RoBERTa_UC
```bash
python ensemble_UC.py --model_type RoBERTa --model_path1 JTCSE_saved_ckpt/JTCSE_RoBERTa/JTCSE_roberta_ckpt_encoder_1/ --model_path2 JTCSE_saved_ckpt/JTCSE_RoBERTa/JTCSE_roberta_ckpt_encoder_2/ --ensemble_UC [your InfoCSE-BERT path]
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 74.57 | 85.73 | 78.17 | 85.78 | 82.73 | 83.74        | 73.52            | 80.61 |

### Eval JTCSE_BERT_UC_D
```bash
python evaluation_D.py --model_name_or_path JTCSE_UCD/JTCSE_BERT_UCD
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 75.23 | 85.46 | 78.50 | 85.50 | 81.55 | 83.02        | 74.24           | 80.50 |

### Eval JTCSE_RoBERTa_UC_D
```bash
python evaluation_D.py --model_name_or_path JTCSE_UCD/JTCSE_RoBERTa_UCD
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg.  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:------------:|:---------------:|:-----:|
| 74.92  | 85.14 | 77.07 | 84.59  | 81.71 | 83.18        | 74.50           | 80.16 |

## Train JTCSE

### Data Preparation

We have prepared the unlabelled training set, located in **data/Wiki_for_JTCSE.txt**; the seven STS test sets are contained in **SentEval**.

### Pre-training Model

The checkpoints we need to use for training JTCSE-BERT and JTCSE-RoBERTa are saved in **Pretrained_encoder/BERT/JTCSE_bert_pretrained_encoder_1/**, **Pretrained_encoder/BERT/JTCSE_bert_pretrained_encoder_2/**, **Pretrained_encoder/RoBERTa/JTCSE_roberta_pretrained_encoder_1/**, and **Pretrained_encoder/RoBERTa/JTCSE_roberta_pretrained_encoder_2/**, respectively, which are RTT Data Augmentation and unsupervised SimCSE trained.

### Train JTCSE_BERT
```bash
python train_jtcse_bert.py
```

### Train TNCSE_RoBERTa
```bash
python train_jtcse_roberta.py
```

#  Zero-shot downstream tasks evaluation

### The performance of the JTCSE and baseline models on 19 multilingual/cross-language semantic similarity tasks on the STS22.V2 test set.
| **Tasks**    | **SimCSE** | **ESimCSE** |   **DiffCSE**   | **InfoCSE**  | **SNCSE** | **WhitenedCSE** |   **RankCSE**   |    **TNCSE**    | **JTCSE** | **JTCSE D** | **JTCSE UCD** |
|:-----------------:|:---------------:|:-----------:|:---------------:|:------------:|:--------------:|:--------------------:|:---------------:|:---------------:|:---------:|:-----------:|:-------------:|
| **ar**       | **38.33**  |    32.48    |      34.94      |    21.08     | 33.58          | 36.08                |    <u>38.16     |      34.75      |   35.16   |    32.77    |     33.15     |
| **avg**      | 32.82           |    36.79    |      34.37      |    28.09     | 23.64          | 32.71                |      38.49      |    **39.33**    | <u>39.21  |    37.76    |     37.82     |
| **de**       | 24.70           |  <u>28.50   |      24.47      |    18.02     | 2.58           | 24.99                |      24.70      |      22.05      |   27.86   |    28.36    |   **28.99**   |
| **de-en**    | 13.13           |    29.80    |      33.63      |   <u>37.03   | 20.73          | 30.33                |    **37.52**    |      33.10      |   36.33   |    30.84    |     34.76     |
| **de-fr**    | 35.93           |    32.68    |    <u>38.29     |     2.44     | 25.42          | 31.45                |      37.81      |      35.41      |   32.52   |  **40.43**  |     35.13     |
| **de-pl**    | 18.82           |    12.78    |      11.30      |    -26.67    | 7.08           | 9.58                 |      5.67       |    **36.71**    |   23.13   |  <u>26.02   |     17.68     |
| **en**       | 59.74           |    61.33    |      61.84      |    55.51     | 54.77          | 60.83                |      62.46      |      61.45      |   62.79   |  **63.59**  |   <u>63.06    |
| **es**       | 49.23           |    52.14    |      55.03      |    49.06     | 39.98          | 55.16                |      59.91      |    <u>61.34     | **63.54** |    57.28    |     57.75     |
| **es-en**    | 30.44           |    37.84    |      36.83      |    38.53     | 21.28          | 34.14                |    **39.37**    |      25.96      | <u>38.77  |    35.18    |     38.00     |
| **es-it**    | 31.48           |    42.50    |      40.91      | <u>44.44 | 22.54          | 31.27                |      42.43      |    **45.70**    |   42.56   |    43.80    |     44.14     |
| **fr**       | 61.55           |    61.31    |      60.06      |    52.95     | 31.47          | 52.96                |      64.85      |    **67.70**    |   64.22   |  <u>66.91   |     65.50     |
| **fr-pl**    | 39.44           |  **50.71**  |      -5.63      |    16.90     | 16.90          | 16.90                |      39.44      |    **50.71**    |   28.17   |    28.17    |     28.17     |
| **it**       | 54.67           |    59.89    |      57.61      |    52.94     | 27.64          | 53.46                |      60.43      |      61.60      | <u>62.37  |    62.05    |   **62.68**   |
| **pl**       | 22.79           |    26.72    |      23.77      |     8.23     | 6.78           | 23.42                |    **31.00**    |      29.67      | <u>30.46  |    26.24    |     26.05     |
| **pl-en**    | 15.44           | <u>36.41  |      30.43      |    29.48     | 28.67          | 22.82                |      34.44      |      29.15      | **37.36** |    27.74    |     30.62     |
| **ru**       | 15.71           |    17.87    |      24.03      |     6.77     | 14.03          | <u>24.59    |      21.70      |    **26.26**    |   23.03   |    20.29    |     20.34     |
| **tr**       | 28.09           |    31.56    |      29.18      |    24.27     | 16.92          | 28.33                |      30.35      |      30.05      | **33.37** |  <u>31.80   |     31.59     |
| **zh**       | 46.42           |    37.76    |    <u>48.78     |    47.06     | 40.12          | 40.45                |    **50.65**    |      39.64      |   48.23   |    41.81    |     42.71     |
| **zh-en**    | 4.82            |    9.87     |      13.14      |  **27.61**   | 15.06          | 11.94                |      12.02      |      16.71      |   15.90   |    16.38    |   <u>20.49    |
| **Avg. Acc** | 32.82           |    36.79    |      34.37      |    28.09     | 23.64          | 32.71                |      38.49      |    **39.33**    | <u>39.21  |    37.76    |     37.82     |


### We report JTCSE and baselines performance on 45 classification tasks.

|                    **Tasks**                    |   **SimCSE**   | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE**  | **WhitenedCSE** | **RankCSE** | **TNCSE** | **JTCSE**  | **JTCSE D** | **JTCSE UCD** |
|:-----------------------------------------------:|:------------------:|:---------------:|:--------------------:|:---------------:|:--------------:|:-------------------:|:---------------:|:-------------:|:--------------:|:-------------:|:-----------------:|
|               **AllegroReviews**                |       24.15        |      23.72      | **25.25**       |      24.13      |     24.38      |     <u>24.94**      |      24.89      |     23.59     |     23.51      |      24.05    |       24.21       |
|                 **AngryTweets**                 |       42.34        |      41.37      | <u>42.54    |      41.68      |   **44.30**    |        41.29        |      42.33      |     41.51     |     40.66      |      40.93    |       41.45       |
|    **tabularContractNLIInclusionOfVerbally**    |       53.96        |    <u>64.75     | 53.96                |      61.87      |   **66.19**    |        48.92        |      52.52      |     56.83     |     58.27      |      54.68    |       52.52       |
|  **tabularContractNLIPermissibleAcquirement**   |       79.21        |    <u>83.15     | 82.58                |      78.65      |     74.16      |      **87.08**      |      81.46      |     82.58     |     82.58      |      82.58    |     <u>83.15      |
|  **tabularContractNLIPermissibleDevelopment**   |       78.68        |    <u>88.24     | 85.29                |      85.29      |     79.41      |      **90.44**      |      83.09      |     79.41     |     85.29      |      82.35    |       85.29       |
|        **CUADAntiAssignmentLegalBench**         |     **84.73**      |      82.76      | 80.89                |      82.00      |     79.10      |        80.46        |      83.11      |   <u>83.70    |     81.66      |      82.51    |       81.14       |
|          **CUADExclusivityLegalBench**          |       66.40        |      70.47      | 63.39                |      63.78      |     64.04      |        68.50        |      62.86      |     64.04     |     70.60      |    **73.10**  |     <u>72.31      |
|     **CUADNoSolicitOfCustomersLegalBench**      |     **84.52**      |    **84.52**    | 79.76                |      76.19      |     77.38      |      **84.52**      |    **84.52**    |     82.14     |   **84.52**    |      83.33    |       83.33       |
|    **CUADPostTerminationServicesLegalBench**    |       60.02        |      57.43      | 55.94                |      60.64      |     59.78      |        58.91        |      57.55      |     57.92     |   **61.14**    |    <u>60.89   |       59.28       |
|   **CUADTerminationForConvenienceLegalBench**   |       80.93        |      79.07      | 79.53                |      83.26      |     67.91      |        77.44        |      80.70      |     77.21     |   **84.65**    |      84.42    |     **84.65**     |
|             **CzechSoMeSentiment**              |       45.75        |      44.34      | 46.57                |      46.01      |     47.58      |      <u>47.62       |      43.90      |   **47.93**   |     47.35      |      47.50    |       47.39       |
|                **GujaratiNews**                 |       40.19        |      40.40      | 40.30                |      39.83      |    <u>40.55    |      **41.18**      |      39.07      |     39.77     |     40.08      |      39.27    |       38.92       |
|                 **HinDialect**                  |       35.92        |      33.35      | 37.50                |    <u>38.67     |   **42.60**    |        35.75        |      31.84      |     38.09     |     35.58      |      34.82    |       34.80       |
|            **IndonesianIdClickbait**            |       54.26        |      54.39      | 54.09                |      54.56      |   **57.57**    |        54.15        |      53.44      |   <u>55.73    |     54.90      |      54.92    |       55.39       |
| **InternationalCitizenshipQuestionsLegalBench** |       57.47        |      57.32      | 56.59                |    **62.01**    |     53.96      |        56.74        |      54.54      |     54.93     |     55.96      |      56.40    |     <u>57.96      |
|                   **KLUE-TC**                   |       21.16        |      20.39      | 21.88                |    <u>22.37     |   **23.34**    |        22.06        |      21.41      |     22.13     |     21.27      |      21.40    |       21.86       |
|                  **Language**                   |       92.56        |      91.40      | 93.83                |    <u>95.04     |   **96.05**    |        92.80        |      93.13      |     93.22     |     92.23      |      92.42    |       93.28       |
|        **LearnedHandsDivorceLegalBench**        |       76.00        |      80.67      | 75.33                |      69.33      |     64.67      |        80.67        |      83.33      |     82.00     |   **85.33**    |    <u>84.67   |       84.00       |
|   **LearnedHandsDomesticViolenceLegalBench**    |       78.16        |      73.56      | 78.74                |      70.69      |     72.41      |        75.86        |      75.29      |     74.71     |   **81.03**    |    <u>79.89   |       77.01       |
|        **LearnedHandsFamilyLegalBench**         |       70.75        |      72.41      | 68.65                |      71.48      |     64.99      |        68.85        |      71.53      |     76.95     |    <u>79.25    |    **79.98**  |       78.08       |
|        **LearnedHandsHousingLegalBench**        |     **74.76**      |    <u>73.34     | 70.70                |      60.30      |     64.21      |        70.12        |      70.61      |     71.88     |     68.41      |      67.38    |       68.95       |
|          **MacedonianTweetSentiment**           |       35.77        |      35.66      | 37.44                |      36.77      |     37.95      |        37.12        |      36.50      |     37.85     |     37.36      |    **38.01**  |     <u>37.98      |
|                 **MarathiNews**                 |       36.24        |      36.68      | 37.33                |      37.47      |     37.52      |      **38.23**      |      35.74      |     37.04     |     37.30      |      37.04    |     <u>37.63      |
|                **MassiveIntent**                |       33.57        |      26.77      | 29.47                |      30.92      |     29.74      |        28.61        |    <u>33.57     |     16.96     |   **37.38**    |      29.46    |       29.37       |
|               **MassiveScenario**               |      <u>35.86      |      28.34      | 31.02                |      34.90      |     31.49      |        30.82        |      34.94      |     20.84     |   **37.50**    |      30.55    |       30.42       |
|             **NorwegianParliament**             |       52.46        |      52.60      | 52.25                |      52.83      |     51.33      |      **53.24**      |    <u>52.89     |     52.35     |     52.88      |      52.69    |       52.64       |
|         **NYSJudicialEthicsLegalBench**         |       47.95        |      47.60      | 45.55                |      48.97      |     49.66      |        44.86        |      47.95      |   **50.68**   |   **50.68**    |      49.66    |       48.29       |
|        **OPP115DataSecurityLegalBench**         |       71.21        |      71.96      | 70.91                |      73.16      |     58.17      |        69.94        |    <u>75.19     |     73.54     |     74.51      |    **75.64**  |       74.06       |
|         **OPP115DoNotTrackLegalBench**          |       81.82        |      86.36      | 78.18                |    <u>90.91     |     80.91      |        80.00        |      81.82      |   <u>90.91    |   **91.82**    |    <u>90.91   |       87.27       |
|    **tabularOPP115InternationalAndSpecific**    |       73.98        |    **80.51**    | 78.88                |    <u>79.08     |     76.73      |        78.27        |      77.04      |     76.33     |     74.18      |      73.37    |       75.00       |
|        **OPP115PolicyChangeLegalBench**         |       87.24        |      87.70      | 86.54                |      88.40      |     83.06      |        88.17        |      84.45      |     86.77     |   **89.79**    |    **89.79**  |     **89.79**     |
| **OPP115ThirdPartySharingCollectionLegalBench** |       65.85        |      65.16      | 65.66                |      64.72      |     60.06      |        62.70        |    **66.48**    |     64.78     |    <u>66.23    |      64.65    |       65.79       |
|      **OPP115UserChoiceControlLegalBench**      |       72.77        |      70.18      | 73.35                |      72.96      |   **74.00**    |        73.42        |    <u>73.67     |     73.03     |     72.64      |      72.90    |       73.61       |
|    **OralArgumentQuestionPurposeLegalBench**    |       22.44        |      21.79      | 21.47                |      19.87      |     24.04      |        23.08        |      21.79      |   **25.32**   |     24.68      |    <u>25.00   |       23.08       |
|                   **PolEmo2**                   |       34.27        |      32.67      | **37.35**       |      35.47      |    <u>36.84    |        34.68        |      34.23      |     33.48     |     33.14      |      31.54    |       33.00       |
|                 **PunjabiNews**                 |       65.92        |      65.16      | 64.84                |      64.27      |   **67.77**    |        62.99        |      63.57      |     65.86     |     63.76      |    <u>66.88   |       66.62       |
|                    **Scala**                    |       50.14        |    <u>50.30     | 49.98                |      50.21      |     50.15      |        50.26        |    **50.37**    |     50.28     |     50.27      |      50.15    |       50.06       |
|           **SCDBPTrainingLegalBench**           |       62.80        |      59.37      | 59.37                |      56.99      |     51.72      |        55.94        |    <u>63.32     |     62.27     |   **64.38**    |      61.74    |       61.74       |
|           **SentimentAnalysisHindi**            |       38.84        |      39.73      | **40.60**       |      39.11      |     39.47      |        39.65        |    <u>40.10     |     38.82     |     38.70      |      38.90    |       39.16       |
|                 **SinhalaNews**                 |       35.97        |      35.80      | 35.46                |    <u>37.22     |   **39.90**    |        35.14        |      34.82      |     34.15     |     33.72      |      33.12    |       34.35       |
|                 **SiswatiNews**                 |       71.25        |      71.63      | 73.13                |    **74.50**    |     73.25      |        73.38        |      72.25      |   <u>73.50    |     71.25      |      71.88    |       71.88       |
|         **SlovakMovieReviewSentiment**          |       52.71        |    **53.85**    | 53.07                |      53.45      |     51.18      |        52.78        |      53.19      |   <u>53.65    |     53.43      |      53.21    |       53.34       |
|                  **TamilNews**                  |       18.25        |      18.21      | 18.25                |    <u>18.97     |   **19.27**    |        18.50        |      17.79      |     18.36     |     18.05      |      17.87    |       17.76       |
|                    **TNews**                    |       16.14        |      16.01      | 16.25                |    **16.76**    |     16.56      |        16.45        |    <u>16.56     |     15.28     |     15.19      |      15.43    |       16.02       |
|                **TweetEmotion**                 |       26.47        |      26.19      | 28.48                |      28.32      |   **29.40**    |      <u>28.56       |      26.08      |     26.72     |     27.23      |      27.15    |       27.55       |
|                 **45 Avg. Acc**                 |       55.37        |      55.49      | 55.07                |      55.42      |     54.11      |        55.22        |      55.23      |     55.22     |   **56.67**    |   <u>56.11    |       56.03       |

### We report JTCSE and baselines performance on 45 retrieval tasks.
| **Metrics**      | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhinenedCSE** | **RankCSE** | **TNCSE** | **JTCSE** | **JTCSE D** | **JTCSE UCD** |
|:-----------------------:|:---------------:|:----------------:|:----------------:|:----------------:|:--------------:|:--------------------:|:----------------:|:--------------:|:---------:|:-----------:|:-------------:|
| **MAP@1**        | 7.16            | 8.43             | 7.60             | 6.70             | 2.77           | 7.57                 | 6.80             | 7.82           | **8.73**  |  **8.73**   |   <u>8.58**   |
| **MAP@5**        | 10.04           | 11.32            | 10.38            | 9.28             | 4.16           | 10.42                | 9.48             | 10.76          | <u>12.01  |  **12.03**  |     11.96     |
| **MAP@10**       | 10.79           | 12.07            | 11.10            | 9.93             | 4.54           | 11.16                | 10.19            | 11.47          | <u>12.75  |  **12.77**  |     12.73     |
| **MRR@1**        | 11.81           | 13.12            | 11.99            | 11.14            | 4.50           | 11.84                | 10.61            | 11.92          | <u>13.59  |  **13.62**  |     13.46     |
| **MRR@5**        | 16.02           | 17.48            | 16.21            | 15.43            | 6.71           | 16.24                | 14.79            | 16.41          |   18.39   |  **18.54**  |   <u>18.40    |
| **MRR@10**       | 16.90           | 18.38            | 17.09            | 16.21            | 7.27           | 17.12                | 15.63            | 17.30          | <u>19.30  |  **19.41**  |     19.27     |
| **NDCG@1**       | 11.62           | 12.83            | 11.76            | 10.94            | 4.42           | 11.66                | 10.50            | 11.68          | <u>13.36  |  **13.38**  |     13.25     |
| **NDCG@5**       | 13.91           | 15.27            | 14.29            | 13.21            | 5.77           | 14.25                | 13.12            | 14.55          | <u>16.32  |  **16.38**  |     16.32     |
| **NDCG@10**      | 15.13           | 16.56            | 15.44            | 14.23            | 6.51           | 15.42                | 14.30            | 15.72          |   17.45   |  <u>17.49   |   **17.52**   |
| **PRECISION@1**  | 11.82           | 13.12            | 11.99            | 11.18            | 4.49           | 11.85                | 10.64            | 11.93          | <u>13.59  |  **13.63**  |     13.46     |
| **PRECISION@5**  | 6.14            | 6.54             | 6.34             | 6.14             | 2.52           | 6.28                 | 5.89             | 6.26           |   7.08    |  **7.15**   |    <u>7.13    |
| **PRECISION@10** | 4.64            | 4.92             | 4.72             | 4.61             | 1.88           | 4.71                 | 4.47             | 4.60           |  <u>5.20  |    5.19     |   **5.24**    |
| **RECALL@1**     | 7.16            | 8.43             | 7.60             | 6.70             | 2.77           | 7.57                 | 6.80             | 7.82           | **8.73**  |  **8.73**   |    <u>8.58    |
| **RECALL@5**     | 14.81           | 16.24            | 14.96            | 13.52            | 6.61           | 15.02                | 13.94            | 15.64          |   17.30   |  <u>17.38   |   **17.49**   |
| **RECALL@10**    | 19.51           | 20.90            | 19.49            | 17.66            | 9.32           | 19.48                | 18.28            | 20.24          |   21.88   |  <u>21.97   |   **22.15**   |
| **45 Avg.**      | 11.83           | 13.04            | 12.06            | 11.12            | 4.95           | 12.04                | 11.03            | 12.27          | <u>13.71  |  **13.76**  |     13.70     |


### We report JTCSE and baselines performance on 14 reranking tasks.
| **Tasks**                     |  **Sim$\sim$**   | **ESim$\sim$** | **Diff$\sim$**  |  **Info$\sim$**   | **SN$\sim$** | **Whitened$\sim$** |  **Rank$\sim$**   | **TN$\sim$** |  **JT$\sim$**   | **JT$\sim$ D**  | **JT$\sim$ UCD** |
|:----------------------------------:|:----------------:|:-------------------:|:---------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:---------------:|:---------------:|:----------------:|
| **Alloprof**                  |      36.67       | 37.81               |      32.07      |       38.52       | 28.20             |       32.04       |       35.68       | 36.25             |    **39.71**    |    <u>39.63     |      38.49       |
| **AskUbuntuDupQuestions**     |      51.88       | 52.28               |      52.08      |       52.83       | 45.53             |       51.60       |    <u>53.76**     | 50.73             |      52.85      |      52.65      |    **54.01**     |
| **CMedQAv2**                  |      13.97       | 14.78               |    <u>15.26     |     **17.21**     | 11.69             |       15.06       |       14.47       | 14.79             |      14.58      |      14.78      |      15.14       |
| **ESCI**                      |    **80.58**     | 80.28               |      80.49      |       80.36       | 78.05             |       80.47       |     <u>80.57      | 79.75             |      80.17      |      80.51      |      80.54       |
| **MindSmall**                 |      28.68       | 28.86               |    <u>29.34     |       29.18       | 26.14             |       28.10       |     **29.45**     | 28.65             |      28.76      |      28.75      |      28.92       |
| **MMarco**                    |       2.48       | 3.77                |      3.64       |     **4.96**      | 2.70              |       4.02        |       3.34        | 2.94              |      4.02       |      4.04       |     <u>4.20      |
| **NamaaMrTydi**               |     <u>39.88     | 37.00               |      34.29      |       26.69       | **41.05**    |       33.48       |       28.62       | 26.33             |      31.42      |      31.38      |      31.89       |
| **RuBQ**                      |    **27.33**     | 24.04               |      24.80      |       23.39       | 20.43             |     <u>25.34      |       22.28       | 18.05             |      23.25      |      23.25      |      23.92       |
| **SciDocsRR**                 |      67.87       | 70.48               |      70.37      |     **71.29**     | 58.90             |       67.63       |       69.89       | 70.51             |      69.85      |      69.59      |     <u>71.23     |
| **StackOverflowDupQuestions** |      39.56       | 40.63               |      42.77      |     **44.21**     | 31.07             |       42.63       |       41.18       | 39.93             |      41.75      |      41.75      |     <u>43.35     |
| **Syntec**                    |      45.65       | 49.60               |      40.28      |       48.99       | 37.39             |       42.25       |       47.51       | 43.86             |    **52.56**    |    <u>50.93     |      50.85       |
| **T2**                        |      55.20       | 55.87               |      56.27      |       56.71       | 52.10             |       56.16       |       55.59       | 55.32             |      56.78      |    **57.34**    |     <u>56.87     |
| **VoyageMMarco**              |      21.60       | 21.41               |      20.90      |     **23.57**     | 16.50             |       21.52       |       21.09       | 20.46             |      22.07      |      21.78      |     <u>22.69     |
| **WebLINXCandidates**         |       7.58       | 9.24                |      7.99       |       9.03        | 6.15              |       7.79        |     <u>9.64**     | 8.82              |      9.25       |      8.71       |    **10.26**     |
| **Avg. MAP**                  |      37.07       | 37.58               |      36.47      |     <u>37.64      | 32.56             |       36.29       |       36.65       | 35.46             |    <u>37.64     |      37.51      |    **38.03**     |

### We report JTCSE and baselines performance on 15 bi-textmining tasks.

| **Tasks**           | **Sim$\sim$** |  **ESim$\sim$**   | **Diff$\sim$** | **Info$\sim$** |  **SN$\sim$**   | **Whitened$\sim$** | **Rank$\sim$** | **TN$\sim$**  | **JT$\sim$** | **JT$\sim$\ D** | **JT$\sim$\ UCD**  |
|:------------------------:|:------------------:|:-----------------:|:-------------------:|:-------------------:|:---------------:|:------------------:|:-------------------:|:-------------:|:------------:|:---------------:|:------------------:|
| **BUCC**            | 0.55               |       1.54        | 0.54                | 0.60                |      0.12       |        0.25        | 0.62                |   **2.58**    |   <u>2.38    |      2.23       |        1.56        |
| **BUCC.v2**         | 3.40               |       4.98        | 3.33                | 4.27                |      1.52       |        2.78        | 4.21                |   **7.24**    |   <u>7.15    |      7.13       |        6.70        |
| **DiaBla**          | 4.08               |       5.55        | 3.71                | 4.36                |      2.07       |        3.56        | 3.80                |   **6.95**    |   <u>6.61    |      6.55       |        4.98        |
| **Flores**          | 4.82               |      <u>5.50      | 4.18                | 3.75                |      2.74       |        3.44        | 5.04                |   **5.56**    |     5.49     |      5.36       |        4.92        |
| **IN22Conv**        | 1.12               |       1.12        | 1.11                | **1.42**       |      1.06       |        1.11        | 1.16                |     1.16      |   <u>1.23    |      1.20       |        1.23        |
| **IN22Gen**         | 2.35               |       2.75        | 2.50                | **3.97**       |      2.01       |        2.67        | 2.95                |     2.78      |     2.98     |      2.89       |      <u>3.09       |
| **LinceMT**         | 15.44              |       15.65       | 15.53               | 15.22               |      4.43       |       16.22        | 14.45               |     16.30     |   <u>16.98   |      16.68      |     **17.10**      |
| **NollySenti**      | 18.76              |       19.78       | 19.01               | 22.33               |      10.12      |       19.44        | 18.95               |   <u>22.35    |  **22.61**   |      21.85      |       22.05        |
| **NorwegianCourts** | 87.46              |       87.82       | 88.04               | 90.42               |      83.77      |       88.73        | 85.82               |     90.75     |  **90.99**   |    <u>90.96     |       90.67        |
| **NTREX**           | 8.70               |       9.85        | 7.82                | 6.81                |      5.08       |        6.98        | 8.96                |   **10.58**   |   <u>10.48   |      10.18      |        9.19        |
| **NusaTranslation** | 45.52              |       45.93       | 44.61               | **50.36**      |    <u>50.31     |       48.33        | 44.13               |     48.85     |    49.33     |      46.60      |       47.14        |
| **Phinc**           | 33.15              |       34.80       | 40.41               | **43.80**      |      27.58      |       41.79        | 38.13               |     41.43     |    41.33     |      39.53      |      <u>42.40      |
| **RomaTales**       | 2.34               |       2.43        | 3.27                | 3.17                |      3.83       |        3.21        | 2.00                |   **4.43**    |     3.51     |     <u>4.11     |        3.75        |
| **Tatoeba**         | 3.25               |       3.56        | 3.27                | 3.61                |      1.74       |        3.21        | 3.43                |   **4.31**    |   <u>4.23    |      4.09       |        4.00        |
| **TbilisiCityHall** | 0.71               |       0.95        | 0.56                | 1.22                |      0.03       |        0.59        | **1.46**       |     1.17      |     1.30     |     <u>1.41     |        1.30        |
| **Avg. F1**         | 15.44              |       16.15       | 15.86               | 17.02               |      13.10      |       16.15        | 15.67               |   <u>17.76    |  **17.77**   |      17.39      |       17.34        |

