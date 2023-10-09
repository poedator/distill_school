import json

with open('results.json', 'r') as file:
    data = json.load(file)

print(json.dumps(data, indent=4, ensure_ascii=False))
