import os
import sys
import argparse
from statistics import mean
from tabulate import tabulate

from utils import check_files, check_submission_folder, evaluate, zip_file

parser = argparse.ArgumentParser(description='Checking file submission format.')
parser.add_argument('-s', '--submission_folder', type=str, required=True, help='Submission folder path.')
parser.add_argument(
    '-p', '--profile',
    default='dev',            # or any default value you prefer
    help='Profile to use for submission.'
)
parser.add_argument('--evaluate', action='store_true', help='Evaluate the submission file.')
parser.add_argument('-g', '--gold_data', type=str, help='Path to the gold data file.', default='')
parser.add_argument('--zip_for_submission', action='store_true', help='Zip the submission file.')
args = parser.parse_args()

submission_folder = args.submission_folder
gold_data_path = args.gold_data
check_status = False
pred_langs = []

print('\n==============')
print('Checklist:')
print('==============')

output = check_submission_folder(submission_folder)
if len(output) == 4:
  checklist, task, pred_langs, check_status = output
  print(tabulate(checklist, headers=['Item', 'Status', 'Comment'], tablefmt='grid'))
elif len(output) == 1:
  checklist = output[0]
  print(tabulate(checklist, headers=['Item', 'Status', 'Comment'], tablefmt='grid'))
  sys.exit(output)

if not check_status:
  sys.exit('\nSome checks failed. Please fix the issues and try again.')
else:
  if args.zip_for_submission:
    print('\nZipping the submission file...')
    zip_file(submission_folder)

  if args.evaluate:
    if not gold_data_path:
      sys.exit('\nGold data folder path is required for evaluation.')
    if not os.path.exists(gold_data_path):
      sys.exit('\nGold data folder not found.')

    print('\nEvaluating prediction files...\n')
    for language in pred_langs:
      gold_file = os.path.join(gold_data_path, f'{language}.csv')
      pred_file = os.path.join(submission_folder, f'pred_{language}.csv')

      if not os.path.exists(gold_file):
        print(f'Gold data file for {language} not found.')
        continue
      
      gold_lines, pred_lines = check_files(gold_file, pred_file)

      if task == 'b':
        emotion_r = evaluate(gold_lines, pred_lines, task)
        print(f'Evaluation scores for {language} track B:\n')
        for emotion, score in emotion_r.items():
          print(f'{emotion} Pearson r: {score}')
        print(f'\nAverage Pearson r: {round(mean(emotion_r.values()), 4)}')
      else:
        eval_scores = evaluate(gold_lines, pred_lines, task)
        for average in eval_scores:
          print(f'Evaluation scores ({average}) for {language} track {task}:')
          print('Precision:', eval_scores[average]['precision'])
          print('Recall:', eval_scores[average]['recall'])
          print('F1 score:', eval_scores[average]['f1'])
          print()