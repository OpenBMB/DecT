
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import *
from transformers.tokenization_utils import SPECIAL_TOKENS_MAP_FILE
import pandas as pd
from openprompt.utils.logging import logger
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

def load_dataset(dataset):
    r"""A dataset loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.
    
    """

    processor = PROCESSORS[dataset.lower()]()
    path = 'datasets/' + dataset.lower()

    train_dataset = None
    valid_dataset = None
    
    try:
        train_dataset = processor.get_train_examples(path)
    except FileNotFoundError:
        logger.warning(f"Has no training dataset in {path}.")
    try:
        valid_dataset = processor.get_dev_examples(path)
    except FileNotFoundError:
        logger.warning(f"Has no validation dataset in {path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {path}.")
    # checking whether donwloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    return train_dataset, valid_dataset, test_dataset, processor
        

class MnlimmProcessor(DataProcessor):
    # TODO Test needed
    dataset_project = {"train": "train",
                        "dev": "dev_mismatched",
                        "test": "dev_mismatched"
                        }
    def __init__(self):
        super().__init__()
        self.labels = ["contradiction", "entailment", "neutral"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.tsv".format(self.dataset_project[split]))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(reader, None)
            for row in reader:
                label, headline, body = self.get_label_id(row[-1]), row[8], row[9]
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(
                    guid=str(0), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
                
        return examples

class MnlimProcessor(DataProcessor):
    # TODO Test needed
    dataset_project = {"train": "train",
                        "dev": "dev_matched",
                        "test": "dev_matched"
                        }
    def __init__(self):
        super().__init__()
        self.labels = ["contradiction", "entailment", "neutral"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.tsv".format(self.dataset_project[split]))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(reader, None)
            for row in reader:
                label, headline, body = self.get_label_id(row[-1]), row[8], row[9]
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(
                    guid=str(0), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
                
        return examples


class AgnewsProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset
    
    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "agnews"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["World", "Sports", "Business", "Tech"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label)-1)
                examples.append(example)
        return examples
    
class DBpediaProcessor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "dbpedia"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 14
        assert len(trainvalid_dataset) == 560000
        assert len(test_dataset) == 70000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["company", "school", "artist", "athlete", "politics", "transportation", "building", "river", "village", "animal", "plant", "album", "film", "book",]

    def get_examples(self, data_dir, split):
        examples = []
        label_file  = open(os.path.join(data_dir,"{}_labels.txt".format(split)),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a+"."
                text_b = ". ".join(text_b)
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(labels[idx]))
                examples.append(example)
        return examples
    

class ImdbProcessor(DataProcessor):
    """
    `IMDB <https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf>`_ is a Movie Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "imdb"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert len(trainvalid_dataset) == 25000
        assert len(test_dataset) == 25000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r') 
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples


    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file  = open(os.path.join(data_dir,dirname,"{}_labels.txt".format('test')),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        return labels

class YahooProcessor(DataProcessor):
    """
    Yahoo! Answers Topic Classification Dataset
    
    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music"
                        ,"Family & Relationships", "Politics & Government"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label)-1)
                examples.append(example)
        return examples


class SST2Processor(DataProcessor):
    """
    `SST-2 <https://nlp.stanford.edu/sentiment/index.html>`_ dataset is a dataset for sentiment analysis. It is a modified version containing only binary labels (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) on top of the original 5-labeled dataset released first in `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank <https://aclanthology.org/D13-1170.pdf>`_

    We use the data released in `Making Pre-trained Language Models Better Few-shot Learners (Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_

    Examples:

    ..  code-block:: python 

        from openprompt.data_utils.lmbff_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SST-2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert processor.get_labels() == ['0','1']
        assert len(train_dataset) == 6920
        assert len(dev_dataset) == 872
        assert len(test_dataset) == 1821
        assert train_dataset[0].text_a == 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films'
        assert train_dataset[0].label == 1

    """
    dataset_project = {"train": "train",
                        "dev": "dev",
                        "test": "dev"
                        }
    def __init__(self):
        super().__init__()
        self.labels = ['0', '1']
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{self.dataset_project[split]}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=self.get_label_id(label))
                examples.append(example)
        return examples


class SnliProcessor(DataProcessor):
    dataset_project = {"train": "snli_1.0_train",
                        "dev": "snli_1.0_dev",
                        "test": "snli_1.0_dev"
                        }
    def __init__(self):
        super().__init__()
        self.labels = ["contradiction", "entailment", "neutral"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(self.dataset_project[split]))
        examples = []
        with open(path)as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                data = json.loads(line)
                text_a = data["sentence1"]
                text_b = data["sentence2"]
                if data["gold_label"] not in self.labels:
                    continue
                label = self.get_label_id(data["gold_label"])
                example = InputExample(
                        guid=str(idx), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples

class YelpProcessor(DataProcessor):
    dataset_project = {"train": "train",
                        "dev": None,
                        "test": "test"
                        }
    def __init__(self):
        super().__init__()
        self.labels = ["1", "2"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(self.dataset_project[split]))
        df = pd.read_csv(path, header=None)
        examples = []
        for idx, (label, text) in enumerate(zip(df[0], df[1])):
            text_a = text
            label = self.get_label_id(str(label))
            example = InputExample(
                    guid=str(idx), text_a=text_a, text_b="", label=label)
            examples.append(example)
        return examples


class RteProcessor(DataProcessor):
    dataset_project = {"train": "train",
                        "dev": "dev",
                        "test": "dev"
                        }
    def __init__(self):
        super().__init__()
        self.labels = ["not_entailment", "entailment"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.tsv".format(self.dataset_project[split]))
        examples = []
        with open(path)as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                line_list = line.strip().split('\t')
                text_a = line_list[-3]
                text_b = line_list[-2]
                label = self.get_label_id(line_list[-1])
                example = InputExample(
                        guid=str(idx), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples

class FewNERDProcessor(DataProcessor):
    """
    `Few-NERD <https://ningding97.github.io/fewnerd/>`_ a large-scale, fine-grained manually annotated named entity recognition dataset

    It was released together with `Few-NERD: Not Only a Few-shot NER Dataset (Ning Ding et al. 2021) <https://arxiv.org/pdf/2105.07464.pdf>`_
    
    Examples:

    ..  code-block:: python

        from openprompt.data_utils.typing_dataset import PROCESSORS

        base_path = "datasets/Typing"

        dataset_name = "FewNERD"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 66
        assert processor.get_labels() == [
            "person-actor", "person-director", "person-artist/author", "person-athlete", "person-politician", "person-scholar", "person-soldier", "person-other",
            "organization-showorganization", "organization-religion", "organization-company", "organization-sportsteam", "organization-education", "organization-government/governmentagency", "organization-media/newspaper", "organization-politicalparty", "organization-sportsleague", "organization-other",
            "location-GPE", "location-road/railway/highway/transit", "location-bodiesofwater", "location-park", "location-mountain", "location-island", "location-other",
            "product-software", "product-food", "product-game", "product-ship", "product-train", "product-airplane", "product-car", "product-weapon", "product-other",
            "building-theater", "building-sportsfacility", "building-airport", "building-hospital", "building-library", "building-hotel", "building-restaurant", "building-other",
            "event-sportsevent", "event-attack/battle/war/militaryconflict", "event-disaster", "event-election", "event-protest", "event-other",
            "art-music", "art-writtenart", "art-film", "art-painting", "art-broadcastprogram", "art-other",
            "other-biologything", "other-chemicalthing", "other-livingthing", "other-astronomything", "other-god", "other-law", "other-award", "other-disease", "other-medical", "other-language", "other-currency", "other-educationaldegree",
        ]
        assert dev_dataset[0].text_a == "The final stage in the development of the Skyfox was the production of a model with tricycle landing gear to better cater for the pilot training market ."
        assert dev_dataset[0].meta["entity"] == "Skyfox"
        assert dev_dataset[0].label == 30
    """
    def __init__(self):
        super().__init__()
        self.labels = [
            "person-actor", "person-director", "person-artist/author", "person-athlete", "person-politician", "person-scholar", "person-soldier", "person-other",
            "organization-showorganization", "organization-religion", "organization-company", "organization-sportsteam", "organization-education", "organization-government/governmentagency", "organization-media/newspaper", "organization-politicalparty", "organization-sportsleague", "organization-other",
            "location-GPE", "location-road/railway/highway/transit", "location-bodiesofwater", "location-park", "location-mountain", "location-island", "location-other",
            "product-software", "product-food", "product-game", "product-ship", "product-train", "product-airplane", "product-car", "product-weapon", "product-other",
            "building-theater", "building-sportsfacility", "building-airport", "building-hospital", "building-library", "building-hotel", "building-restaurant", "building-other",
            "event-sportsevent", "event-attack/battle/war/militaryconflict", "event-disaster", "event-election", "event-protest", "event-other",
            "art-music", "art-writtenart", "art-film", "art-painting", "art-broadcastprogram", "art-other",
            "other-biologything", "other-chemicalthing", "other-livingthing", "other-astronomything", "other-god", "other-law", "other-award", "other-disease", "other-medical", "other-language", "other-currency", "other-educationaldegree",
        ]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "supervised/{}.txt".format(split))
        with open(path, encoding='utf8') as f:
            data = FewNERDProcessor.load_data(f)

            examples = []

            for idx, (xs, ys, spans) in enumerate(data):
                for span in spans:
                    text_a = " ".join(xs)
                    meta = {
                        "entity": " ".join(xs[span[0]: span[1]+1])
                    }
                    example = InputExample(guid=str(idx), text_a=text_a, meta=meta, label=self.get_label_id(ys[span[0]][2:]))
                    examples.append(example)
        
        return examples

    @staticmethod
    def load_data(file):
        data = []
        xs = []
        ys = []
        spans = []

        for line in file.readlines():
            pair = line.split()
            if pair == []:
                if xs != []:
                    data.append((xs, ys, spans))
                xs = []
                ys = []
                spans = []
            else:
                xs.append(pair[0])

                tag = pair[-1]
                if tag != 'O':
                    if len(ys) == 0 or tag != ys[-1][2:]:
                        tag = 'B-' + tag
                        spans.append([len(ys), len(ys)])
                    else:
                        tag = 'I-' + tag
                        spans[-1][-1] = len(ys)
                ys.append(tag)
        return data

PROCESSORS = {
    "agnews": AgnewsProcessor,
    "dbpedia": DBpediaProcessor,
    "imdb": ImdbProcessor,
    "sst2": SST2Processor,
    "mnlim": MnlimProcessor,
    "mnlimm": MnlimmProcessor,
    "yahoo": YahooProcessor,
    "yelp": YelpProcessor,
    "snli": SnliProcessor,
    "rte": RteProcessor,
    "fewnerd": FewNERDProcessor,
}