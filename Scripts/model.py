import pandas as pd
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel, MultiLabelClassificationArgs
from sys import argv
from sklearn.metrics import accuracy_score, f1_score
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(torch.cuda.is_available())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

model_type = argv[1]
model_name = argv[2]
total_epochs = int(argv[3])

print(f'begin-{model_type}-{model_name}')

print(argv[1], argv[2], argv[3])

train_df = pd.read_csv(f'../datasets/train.csv')
eval_df = pd.read_csv(f'../datasets/test.csv')

#train_df = train_df.iloc[:10]
#eval_df = eval_df.iloc[:5]

#print(type(train_df["labels"][0]))

print(type(train_df["labels"][1]))

def str_to_list(text):
    text = list(map(int,text.strip('][').split(', ')))
    return text
    
train_df['labels'] = train_df['labels'].apply(str_to_list)
eval_df['labels'] = eval_df['labels'].apply(str_to_list)

print(train_df.head())

print(type(train_df["labels"][1]))

cuda_available = torch.cuda.is_available()

### START TRAINING

currently_epoch = 0

train_args = {
    'output_dir': f'../outputs/{model_type}-{model_name}-outputs/',

    'fp16': False,

    'max_seq_length': 512,
    'num_train_epochs': total_epochs,
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

model = MultiLabelClassificationModel(model_type, model_name, num_labels=4, args=train_args, use_cuda=cuda_available)

acc_score,f1_label = 0,0

#def label_num(y, ln):
#    return[elem[ln] for elem in y]

model.train_model(train_df,
 #f1=sklearn.metrics.f1_score, 
 #acc=sklearn.metrics.accuracy_score
 )
result, model_outputs, wrong_predictions = model.eval_model(eval_df, 
#f1=sklearn.metrics.f1_score, 
#acc=sklearn.metrics.accuracy_score
)
print("model outputs")
print(model_outputs)



y_test = [eval_df["labels"][idx] for idx in eval_df.index]


def label_num(y, label):
    return [l[label] for l in y]

def fix_y(y):
    y = y.round()
    return y
    
y_predict = model_outputs
    
y_predict = y_predict.round()
print("Y_predict", y_predict)

acc_score_IE = accuracy_score(label_num(y_predict, 0), label_num(y_test, 0))
acc_score_NS = accuracy_score(label_num(y_predict, 1), label_num(y_test, 1))
acc_score_TF = accuracy_score(label_num(y_predict, 2), label_num(y_test, 2))
acc_score_JP = accuracy_score(label_num(y_predict, 3), label_num(y_test, 3))

f1_IE = f1_score(label_num(y_predict, 0), label_num(y_test, 0))
f1_NS = f1_score(label_num(y_predict, 1), label_num(y_test, 1))
f1_TF = f1_score(label_num(y_predict, 2), label_num(y_test, 2))
f1_JP = f1_score(label_num(y_predict, 3), label_num(y_test, 3))

acc_score = accuracy_score(y_predict, y_test)

print(f"acc:{acc_score}")

with open(f"../outputs/{model_type}-{model_name}-outputs/results_acc.txt", "w") as results_file:

    results_file.write(f" acc_score_IE: {acc_score_IE}\n")
    results_file.write(f" f1_IE score: {f1_IE}\n") 
    results_file.write(f" acc_score_NS: {acc_score_NS}\n")
    results_file.write(f" f1_NS score: {f1_NS}\n") 
    results_file.write(f" acc_score_TF: {acc_score_TF}\n")
    results_file.write(f" f1_TF score: {f1_TF}\n") 
    results_file.write(f" acc_score_JP: {acc_score_JP}\n")
    results_file.write(f" f1_JP score: {f1_JP}\n") 

    results_file.write('\n')
    results_file.write(f'acc_complete: {acc_score}\n')
    acc_prom = (acc_score_IE + acc_score_NS + acc_score_TF + acc_score_JP)/4
    results_file.write(f'acc_prom: {acc_prom}\n')
    f1_prom = (f1_IE + f1_NS + f1_TF + f1_JP)/4
    results_file.write(f'f1 prom: {f1_prom}\n')

currently_epoch += 1

print(f'end-{model_type}-{model_name}')

