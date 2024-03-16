import json


file_path = ''
org_fid_trainingfile_path = ''

results = []

with open(file_path, "r") as f:
    for line in f:
        per_result = line.strip().split('\t')
        try:
            results.append(per_result[1])
        except:
            results.append('')


with open(org_fid_trainingfile_path) as f:
    dataset = json.load(f)


print(len(dataset), len(results))
# exit()

processed_dataset = []
for i, data in enumerate(dataset):
    
    print(data['question'], results[i])
    data['answers'] = [results[i]]
    processed_dataset.append(data)


with open(org_fid_trainingfile_path + '_cot.json', 'w') as outfile:
    json.dump(processed_dataset, outfile, indent=4)
