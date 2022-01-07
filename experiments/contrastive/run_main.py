# python -m contrastive.run_main
import os

command = '''
python -m contrastive.main \
    --model_name_or_path "bert-base-uncased" \
    --task_name "sst-2" \
    --data_dir "/home/lxuechen_stanford_edu/software/swissknife/experiments/contrastive/data-glue-format/orig" \
    --output_dir "/nlp/scr/lxuechen/contrastive/test"
'''

os.system(command)
