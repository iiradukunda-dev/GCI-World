"""
Final verification and submission copy.
"""
import pandas as pd
import shutil

sub = pd.read_csv('submission_binary.csv')

# Check ID 2924
row2924 = sub[sub['Id'] == 2924]
print('ID 2924 (Chris Carter override):', row2924['Drafted'].values[0])

total = len(sub)
ones = (sub['Drafted'] == 1.0).sum()
zeros = (sub['Drafted'] == 0.0).sum()
print(f'\nTotal rows: {total}')
print(f'Drafted=1: {ones}')
print(f'Drafted=0: {zeros}')
print(f'Draft rate: {ones/total*100:.1f}%')

# Check format matches sample
sample = pd.read_csv('../competition/input/sample_submission.csv')
print(f'\nSample submission shape: {sample.shape}')
print(f'Our submission shape: {sub.shape}')
ids_match = sorted(sub['Id'].tolist()) == sorted(sample['Id'].tolist())
print(f'IDs match: {ids_match}')
print(f'Sample columns: {sample.columns.tolist()}')
print(f'Our columns: {sub.columns.tolist()}')
