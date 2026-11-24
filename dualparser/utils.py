import json
import pickle


def cache_to_file(log_tuples, cached_file):
    with open(cached_file, "w") as file:
        for log, index in log_tuples:
            file.write(f"{index} {log}\n")


def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print("No file:", file_path)
        return None
    except Exception as e:
        print("Load Error:", str(e))
        return None


def save_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print("Save error:", str(e))


def load_tuple_list(file_path):
    with open(file_path, 'r') as f:
        tuple_list = []
        for line in f:
            index, value = line.strip().split(' ', 1)
            tuple_list.append((value, int(index)))
        return tuple_list


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]