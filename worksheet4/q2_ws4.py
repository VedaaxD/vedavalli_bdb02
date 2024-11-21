from logging import exception
import JSONProcessor as jp
from JSONProcessor import load_json, man_of_the_match, save_updated_json
def main():
    file_path='/home/ibab/worksheet4/cricketers.json'
    try:
        data = load_json('/home/ibab/worksheet4/cricketers.json')
        print(f"Original JSON data:{data}")
        updated_data = man_of_the_match(data)
        print(f"Updated JSON data: {updated_data}")
        saved_update= save_updated_json(data,file_path)
        print(f"Data is successfully updated {file_path}.")

    except Exception as e:
        print(f"Error has occured. {e} ")
if __name__=="__main__":
    main()