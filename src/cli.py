import os
import sys
sys.path.append(".")

import argparse
import csv
from re import template
from process_data import load_dataset
from dect_verbalizer import DecTVerbalizer
from dect_trainer import DecTRunner
from openprompt.prompts import ManualTemplate
from openprompt.pipeline_base import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import logger
from openprompt.plms import load_plm
from openprompt.data_utils.utils import InputExample
from openprompt.utils.cuda import model_to_device



parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='roberta')
parser.add_argument("--model_name_or_path", default='roberta-large')
parser.add_argument("--shot", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--verbalizer", type=str, default='proto')
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--dataset", type=str, default='sst2')
parser.add_argument("--max_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--proto_dim", type=int, default=128)
parser.add_argument("--model_logits_weight", type=float, default=1)
parser.add_argument("--lr", default=0.01, type=float)
args = parser.parse_args()




def build_dataloader(dataset, template, verbalizer, tokenizer, tokenizer_wrapper_class, batch_size):
    dataloader = PromptDataLoader(
        dataset = dataset, 
        template = template, 
        verbalizer = verbalizer, 
        tokenizer = tokenizer, 
        tokenizer_wrapper_class=tokenizer_wrapper_class, 
        batch_size = batch_size,
    )

    return dataloader



def main():

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(args.dataset)

    # main
    sampler = FewShotSampler(
                num_examples_per_label = args.shot,
                also_sample_dev = True,
                num_examples_per_label_dev = args.shot)

    train_sampled_dataset, valid_sampled_dataset = sampler(
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        seed = args.seed
    )

    set_seed(123)

    plm, tokenizer, model_config, plm_wrapper_class = load_plm(args.model, args.model_name_or_path)

    # define template and verbalizer
    # define prompt
    template = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/{args.dataset}/manual_template.txt", choice=args.template_id)
    verbalizer = DecTVerbalizer(
        tokenizer=tokenizer, 
        classes=Processor.labels, 
        hidden_size=model_config.hidden_size, 
        lr=args.lr, 
        mid_dim=args.proto_dim, 
        epochs=args.max_epochs, 
        model_logits_weight=args.model_logits_weight).from_file(f"scripts/{args.dataset}/manual_verbalizer.json")
    # load prompt’s pipeline model
    prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm = True)
            
    # process data and get data_loader
    train_dataloader = build_dataloader(train_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if train_dataset else None
    valid_dataloader = build_dataloader(valid_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if test_dataset else None

    calibrate_dataloader =  PromptDataLoader(
                            dataset = [InputExample(guid=str(0), text_a="", text_b="", meta={"entity": "It"}, label=0)], 
                            template = template, 
                            tokenizer = tokenizer, 
                            tokenizer_wrapper_class=plm_wrapper_class
                        )

    runner = DecTRunner(model = prompt_model,
                        train_dataloader = train_dataloader,
                        valid_dataloader = valid_dataloader,
                        test_dataloader = test_dataloader,
                        calibrate_dataloader = calibrate_dataloader,
                        id2label = Processor.id2label
            )                                   

        
    res = runner.run()
    '''
    if config.proto_verbalizer.multi_verb == 'manual':
        name_list = [config.dataset.name, config.plm.model_path, config.sampling_from_train.num_examples_per_label]
        filename = 'experiments/results-wosr/' + '-'.join([str(e) for e in name_list]) + '.csv'
    else:
        name_list = [config.dataset.name, config.plm.model_path, config.sampling_from_train.num_examples_per_label, config.manual_template.choice]
        filename = 'experiments/results-temp/' + '-'.join([str(e) for e in name_list]) + '.csv'
    with open(filename, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([res*100])
    '''
    return res
    


if __name__ == "__main__":
    main()
    
