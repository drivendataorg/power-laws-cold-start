import os
import shutil
import json

def main():
    with open('taylor.json', 'r') as f:
        confs = json.load(f)
    for window in confs:
        for input_days in confs[window]:
            model_paths = confs[window][input_days]
            for model_path in model_paths:
                conf_path = model_path.replace('.h5', '.json')
                new_conf_path = os.path.join(
                    '../final_solution/tailor_confs', window, input_days,
                    '%s.json' % os.path.basename(os.path.dirname(conf_path)))
                if not os.path.exists(os.path.dirname(new_conf_path)):
                    os.makedirs(os.path.dirname(new_conf_path))
                shutil.copy(conf_path, new_conf_path)

    with open('seq2seq.json', 'r') as f:
        confs = json.load(f)
    for input_days in confs:
        for is_working in confs[input_days]:
            model_paths = confs[input_days][is_working]
            for model_path in model_paths:
                conf_path = model_path.replace('.h5', '.json')
                new_conf_path = os.path.join(
                    '../final_solution/seq2seq_confs', input_days, is_working,
                    '%s.json' % os.path.basename(os.path.dirname(conf_path)))
                if not os.path.exists(os.path.dirname(new_conf_path)):
                    os.makedirs(os.path.dirname(new_conf_path))
                shutil.copy(conf_path, new_conf_path)

if __name__ == "__main__":
    main()
