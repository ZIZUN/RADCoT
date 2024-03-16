import openai
import json 
from time import sleep


openai.api_key = ""

model = "text-davinci-003"

dataset_name = '19_bm25'

query_path = ''
write_path = ''


with open(query_path) as f:
    dataset = json.load(f)

fw = open(write_path, 'a')



results = {}
count = 0

datasetlen = len(dataset)
print(f'{datasetlen}')

while True:

    data = dataset[count]

    question = 'Answer the following question \"' + data['question'] + '\".\nGive the rationale before answering.\n'

    try:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question,
        max_tokens=150,
        temperature=0
        )
        answer = response
    except:
        sleep(30)
        continue

    print(f'{count}/{datasetlen}') # logging
    fw.write(str(count) + "\t" + response["choices"][0]["text"].replace("\n", " ", 20) + '\n')
    count += 1      


