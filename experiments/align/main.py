import json

import fire


def main(task="redteam"):
    if task == "rtp":
        with open('prompts.jsonl') as f:
            lines = f.readlines()

        max_num = 5000
        out = []
        for i, line in enumerate(lines):
            line = json.loads(line)
            try:
                if line['prompt']['severe_toxicity'] is None:
                    line['prompt']['severe_toxicity'] = 0.
                out.append(line)
            except KeyError:
                pass
            if i >= max_num:
                break

        out = sorted(out, key=lambda x: x['prompt']['severe_toxicity'], reverse=True)
        for outi in out[:50]:
            print(outi['prompt']['text'])
    elif task == "redteam":
        with open("./train.jsonl") as f:
            lines = f.readlines()

        print(len(lines))
        breakpoint()
        for line in lines:
            this = json.loads(line)
            print(this)
            print(this.keys())
            print(this['chosen'])
            print('------------------------')
            print(this['rejected'])
            breakpoint()


if __name__ == '__main__':
    fire.Fire(main)
