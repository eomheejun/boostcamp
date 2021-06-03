import logging
import os
import sys
from datasets import load_metric, load_from_disk
import torch

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer,Trainer
from transformers import ElectraModel, ElectraTokenizer

from transformers import EarlyStoppingCallback
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
import wandb
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, 
        #TrainingArguments
        )
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    training_args = TrainingArguments(
        output_dir = './models/train_dataset',
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=300,        
        dataloader_num_workers=4,
        num_train_epochs=10,
        save_total_limit=2,
        fp16 = True,        
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 32 ,       
        #save_strategy = 'steps',
        #save_steps=10,      
        #logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', #-> 'IntervalStrategy.steps
        eval_steps=100,        
        load_best_model_at_end = True,
        metric_for_best_model = 'exact_match',
        #do_eval=True
        #do_eval=False
        #lr_scheduler_type="constant",
        gradient_accumulation_steps = 4,
        run_name = 'lr_3e-5_batch16_GradA4_maxseq384_one',
        #label_smoothing_factor=0.5
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    
    josa=list(set(['이','가','께서','에서','을','를','에','에게','께','에서','에게서','한테서','에서', '로','으로','으로서','의','과','와','하고','보다', '처럼', '만큼', '같이','아','야','이여','여','이시여','시여', '만','도','은','는','이','가','조차', '마저', '까지', '부터', '이나','나', '이나마','나마','이라도','라도', '이야','야', '이라야','라야'])) 
    special_tokens_dict = {'additional_special_tokens': josa}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    ) 
    model.resize_token_embeddings(len(tokenizer))

    # 이 파일은 mrc 학습을 하기 때문에, run_sparse_embedding실행을 하지 않도록 설정했음.
    # 원래는 sparse embedding하고 mrc 학습하도록 되어 있었는데, inference 에서 해도 되니까 안함.    
    if data_args.train_retrieval:
        run_sparse_embedding()
   
    run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_embedding():
    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="./data",
                                context_path="wikipedia_documents.json"
                                #context_path="all_wikipedia_documents.json"
                                )
    retriever.get_sparse_embedding()


def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    wandb.login()
    
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    '''
    if training_args.do_train:
        column_names = datasets["train"].column_names
        # ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']
    else:
        column_names = datasets["validation"].column_names
        # ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']
    '''
    # 어차피 같은 값이라서 하나만 써도 됨.
    column_names = datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)

    # Training preprocessing
    def prepare_train_features(examples):
        # 토크나이징 + 정답 start, end 토큰 위치 모두 겸비한 input data 만들기!

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            #padding="max_length",
            #is_split_into_words=True ,           
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # truncation 으로 구분된 동일 context 에 대한 여러 결과물에 대해, 원래 어느 context에 소속되는지 저장하는 값.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # 모듵 tokeninzed된 결과물에 대해, 각 토큰의 원래 문장에서의 위치값(시작, 끝) 정보를 담고 있음.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # sequence_ids -> cls, 질문, cls, 문단, end 구분 기능 -> [None, 0, 0, ..., 0, None, 1, 1, ...., 1, None] 
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            # 지금 보고있는 i 번째 tokenized 결과가 어느 context로 만들어졌는지 그 번호 찾음. 
            sample_index = sample_mapping[i]
            
            answers = examples[answer_column_name][sample_index]
            # print(answers)
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = datasets["train"]

    # Create train feature from dataset
    # default -> batch_size: Optional[int] = 1000,
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )


    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            #padding="max_length",
            #is_split_into_words=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            # offset_mapping : 각 토큰의 (시작위치, 끝 위치) 정보를 담고 있는데, 
            # query 토큰들의 (시작위치, 끝 위치) 정보를 None으로 바꾸는 과정
            # 왜? validation 할 때, output 으로 start logit과 end logit을 받게 된다. 
            # 이때 해당 인덱스를 query 가 아닌 passage에서 찾기 위함.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature Creation
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    
    # pad_to_multiple_of : mixed precision을 사용할 때, 텐서 사이즈가 8의 배수일때 더 효과적이다.'
    # 따라서,(Funnel Transformer? 뭔지 모르겠지만 이건 32로 세팅) 8로 세팅해서 max_length을 조절 하게 된다. 
    # 근데 이미 tokeneizer가 max_length를 384로 처리하고 있어서 작동 안할 듯.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
       # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions        
        else:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    metric_key_prefix = 'eval'   
    def compute_metrics(p: EvalPrediction):
        before_prefix_metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
        metrics ={f'{metric_key_prefix}_{k}':v for k,v in before_prefix_metrics.items()}        
        return metrics
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience = 10, early_stopping_threshold = 0.2)

    # QuestionAnsweringTrainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
   
    # Training
 
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    
    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")
            
    wandb.finish() 

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
