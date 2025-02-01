import os
import sys
import zipfile
import numpy as np
import scipy.stats as ss
from sklearn.metrics import f1_score, recall_score, precision_score

import warnings
warnings.filterwarnings('ignore')

def check_if_language_is_supported(language):
  if language in open('langs.txt').read().splitlines():
    return True

def check_submission_folder(folder_path):
  checklist = []

  folder = folder_path.split('/')[-1]
  if not (os.path.exists(folder_path) or os.path.isdir(folder_path)):
    checklist.append(['Submission folder.', 'Fail', 'Not found or invalid.'])
    return [checklist]
  else:
    checklist.append(['Submission folder.', 'Pass', f'Found valid folder: {folder}'])

  if not folder.startswith('track_'):
    checklist.append(['Submission folder.', 'Fail', 'Folder name should start with "track_".'])
  else:
    checklist.append(['Submission folder.', 'Pass', f'Folder name: {folder}, starts with "track_"'])

  task = folder.split('_')[-1] if '_' in folder else ''
  if not task:
    checklist.append(['Task name.', 'Fail', 'Task name not found in the folder name. Folder name should have the following format: "track_a" for task a.'])
  else:
    if not task in ['a', 'b', 'c']:
      checklist.append([f'Task name.', 'Fail', 'The task name must be either "a", "b", or "c" after "track_".'])
    else:
      checklist.append([f'Task name.', 'Pass', f'Task: {task.upper()}'])

  prediction_files = os.listdir(folder_path)

  if not prediction_files:
    checklist.append(['Prediction files.', 'Fail', 'No prediction file found.'])
    return [checklist]
  else:
    checklist.append(['Prediction files.', 'Pass', f'Found {len(prediction_files)} prediction files: {", ".join(prediction_files)}'])

  prediction_files = [f for f in prediction_files if f.startswith('pred_') and f.endswith('.csv')]

  pred_langs = []
  for file_name in prediction_files:
    if not (file_name.endswith('.csv') and file_name.startswith('pred_')):
      checklist.append(['File format.', 'Fail', f'{file_name}: Should be a CSV file and start with "pred_".'])
    else:
      try:
        lang = file_name.split('_')[-1].split('.')[0]
        if not check_if_language_is_supported(lang):
          checklist.append(['Language code.', 'Fail', f'Language code "{lang}" not supported.'])
        else:
          pred_langs.append(lang)
      except IndexError:
        checklist.append(['Language code.', 'Fail', 'Language code not found after "pred_".'])

  if len(pred_langs) == len(prediction_files):
    checklist.append(['Prediction files.', 'Pass', 'All prediction files are in the correct format.'])

  check_status = True
  if 'Fail' not in [c[1] for c in checklist]:
    checklist.append(['Submission format check.', 'Pass', 'All checks passed.'])
  else:
    checklist.append(['Submission format check.', 'Fail', 'Some checks failed.'])
    check_status = False

  return [checklist, task, pred_langs, check_status]

def are_lists_equal(list1, list2):
  return [x.lower() for x in list1] == [x.lower() for x in list2]

def check_files(gold, pred):
  gold_lines = open(gold).read().splitlines()
  pred_lines = open(pred).read().splitlines()

  valid_header = gold_lines[0].split(',')
  pred_header = pred_lines[0].split(',')

  if not are_lists_equal(valid_header, pred_header):
    sys.exit(f'Invalid submission header: {pred_header}. Your submission should have an "id" column first and {len(valid_header) - 1} columns for emotions: {valid_header[1:]}, in this order.')

  if len(gold_lines) != len(pred_lines):
    sys.exit(f'Predictions ({os.path.basename(pred)}) and gold data have different number of lines.')

  print('\nFiles format checked successfully\n')

  return gold_lines, pred_lines

def populate_data_dict(gold_lines, pred_lines, label_range):
  data_dic = {}
  
  for row in gold_lines[1:]:
    parts = row.split(',')
    data_dic[parts[0]] = [tuple(map(int, parts[1:label_range]))]
  
  for row in pred_lines[1:]:
    parts = row.split(',')
    if parts[0] in data_dic:
      try:
        data_dic[parts[0]].append(tuple(map(int, parts[1:label_range])))
      except ValueError:
        data_dic[parts[0]].append((0,) * (label_range - 1))
    else:
      sys.exit(f'Invalid tweet id: {parts[0]} found in submitted predictions file.')

  return data_dic

def compute_scores(data_dic):
  gold_scores = [labels[0] for labels in data_dic.values() if len(labels) == 2]
  pred_scores = [labels[1] for labels in data_dic.values() if len(labels) == 2]

  if len(gold_scores) != len(pred_scores):
    sys.exit(f"Error: Mismatch in number of valid predictions and gold labels.")

  return np.array(gold_scores), np.array(pred_scores)

def evaluate(gold_lines, pred_lines, task):
  if task in ['a', 'c']:
    data_dic = populate_data_dict(gold_lines, pred_lines, 8)

    y_true, y_pred = compute_scores(data_dic)

    eval_scores = {}
    for average in ['micro', 'macro']:
      recall = recall_score(y_true, y_pred, average=average)
      precision = precision_score(y_true, y_pred, average=average)
      f1 = f1_score(y_true, y_pred, average=average)

      eval_scores[average] = {'recall': round(recall, 4), 'precision': round(precision, 4), 'f1': round(f1, 4)}

    return eval_scores

  elif task == 'b':
    emotions = gold_lines[0].split(',')[1:]
    data_dic = populate_data_dict(gold_lines, pred_lines, len(emotions) + 1)

    emotions_data_dict = {emotion: {'gold': [], 'pred': []} for emotion in emotions}
    for gold_line, pred_line in zip([d[0] for d in data_dic.values()], [d[1] for d in data_dic.values()]):
      for i, emotion in enumerate(emotions):
        emotions_data_dict[emotion]['gold'].append(float(gold_line[i]))
        emotions_data_dict[emotion]['pred'].append(float(pred_line[i]))

    emotion_r = {
      emotion: round(ss.pearsonr(emotions_data_dict[emotion]['gold'], emotions_data_dict[emotion]['pred'])[0], 4)
        for emotion in emotions
    }

    return emotion_r
  
  else:
    sys.exit('Invalid task name. Task name must be either "a", "b", or "c".')

def zip_file(submission_file):
  zip_file_name = f'{submission_file}.zip'

  with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    zipf.write(submission_file, os.path.basename(submission_file))

  print(f'Zipped folder: {zip_file_name} is ready for upload in the codabench submission page.')