import pandas as pd
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel, MultiLabelClassificationArgs
from sys import argv
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(torch.cuda.is_available())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

model_type = argv[1]
model_name = argv[2]


print(f'begin-{model_type}-{model_name}')

print(argv[1], argv[2])


eval_df = pd.read_csv(f'../datasets/Final_Suicide_Notes.csv')

print(eval_df.head())

cuda_available = torch.cuda.is_available()


train_args = {
    'output_dir': f'../outputs/{model_type}-{model_name}-outputs/',

    'fp16': False,

    'max_seq_length': 512,
    'num_train_epochs': 5,
    'train_batch_size': 3,
    'eval_batch_size': 3,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-5,
    'save_steps': -1,

#         'wandb_project': 'ag-news-transformers-comparison',
#         'wandb_kwargs': {'name': f'{model_type}-{model_name}-{label}'},
#    'evaluate_during_training': True,
#    'evaluate_during_training_steps': 11987,
    'reprocess_input_data': False,
    "save_model_every_epoch": False,
    'overwrite_output_dir': True,
    'no_cache': True,

    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,

    'use_early_stopping': True,
    'early_stopping_patience': 3,
    'manual_seed': 4,
    #'best_model_dir': f'outputs/{model_type}-{model_name}-outputs/best/',
}



print(model_type)
print(model_name)

model = MultiLabelClassificationModel(model_type, "../outputs/"+model_type+"-"+model_name + "-outputs", num_labels=4, args=train_args, use_cuda=cuda_available)

X_test = [eval_df["text"][idx] for idx in eval_df.index]

results, raw_outputs = model.predict(X_test)
#print(results, raw_outputs)

ro = []
for elem in raw_outputs:
    l = []
    for e in elem:
        l.append(e)
    ro.append(l)

d = {
    "results": results,
    "raw_outputs": ro
}
with open(f'../outputs/{model_type}-{model_name}-outputs/suicide_results.json', 'w') as suicide_notes:
    json.dump(d, suicide_notes)
