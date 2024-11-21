import json
from textwrap import indent
def load_json(path):
    try:
        with open('/home/ibab/worksheet4/cricketers.json','r') as file:
            data=json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file is not found at the given path")
    except Exception as e:
        print(e)
def print_json(data):
    json.dumps(data, indent=4)
def man_of_the_match(data):
    max_score=max(player["player_score"] for player in data)
    for player in data:
        player["man_of_the_match"] = player["player_score"] == max_score
    return data
def save_updated_json(data,file_path):
    with open('/home/ibab/worksheet4/cricketers.json','w') as file:
        json.dump(data,file,indent=4)