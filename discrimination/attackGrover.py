
# def _flatten_and_tokenize_metadata(encoder, item):
#     """
#     Turn the article into tokens
#     :param item: Contains things that need to be tokenized
#     fields are ['domain', 'date', 'authors', 'title', 'article', 'summary']
#     :return: dict
#     """
#     metadata = []
#     for key in ['domain', 'date', 'authors', 'title', 'article']:
#         val = item.get(key, None)
#         if val is not None:
#             metadata.append(encoder.__dict__[f'begin_{key}'])
#             metadata.extend(encoder.encode(val))
#             metadata.append(encoder.__dict__[f'end_{key}'])
#     return metadata


# def main(_):
#     LABEL_LIST = ['machine', 'human']
#     LABEL_INV_MAP = {label: i for i, label in enumerate(LABEL_LIST)}

#     tf.logging.set_verbosity(tf.logging.INFO)

#     # These lines of code are just to check if we've already saved something into the directory
#     if tf.gfile.Exists(FLAGS.output_dir):
#         print(f"The output directory {FLAGS.output_dir} exists!")
#         if FLAGS.do_train:
#             print("EXITING BECAUSE DO_TRAIN is true", flush=True)
#             return
#         for split in ['val', 'test']:
#             if tf.gfile.Exists(os.path.join(FLAGS.output_dir, f'{split}-probs.npy')) and getattr(FLAGS,
#                                                                                                  f'predict_{split}'):
#                 print(f"EXITING BECAUSE {split}-probs.npy exists", flush=True)
#                 return
#         # Double check to see if it has trained!
#         if not tf.gfile.Exists(os.path.join(FLAGS.output_dir, 'checkpoint')):
#             print("EXITING BECAUSE NO CHECKPOINT.", flush=True)
#             return
#         stuff = {}
#         with tf.gfile.Open(os.path.join(FLAGS.output_dir, 'checkpoint'), 'r') as f:
#             # model_checkpoint_path: "model.ckpt-0"
#             # all_model_checkpoint_paths: "model.ckpt-0"
#             for l in f:
#                 key, val = l.strip().split(': ', 1)
#                 stuff[key] = val.strip('"')
#         if stuff['model_checkpoint_path'] == 'model.ckpt-0':
#             print("EXITING BECAUSE IT LOOKS LIKE NOTHING TRAINED", flush=True)
#             return


#     elif not FLAGS.do_train:
#         print("EXITING BECAUSE DO_TRAIN IS FALSE AND PATH DOESNT EXIST")
#         return
#     else:
#         tf.gfile.MakeDirs(FLAGS.output_dir)

#     news_config = GroverConfig.from_json_file(FLAGS.config_file)

#     # TODO might have to change this
#     encoder = get_encoder()
#     examples = {'train': [], 'val': [], 'test': []}
#     np.random.seed(123456)
#     tf.logging.info("*** Parsing files ***")
#     with tf.gfile.Open(FLAGS.input_data, "r") as f:
#         for l in f:
#             item = json.loads(l)

#             # This little hack is because we don't want to tokenize the article twice
#             context_ids = _flatten_and_tokenize_metadata(encoder=encoder, item=item)
#             examples[item['split']].append({
#                 'info': item,
#                 'ids': context_ids,
#                 'label': item['label'],
#             })
#             assert item['label'] in LABEL_INV_MAP

#     additional_data = {'machine': [], 'human': []}
#     if FLAGS.additional_data is not None:
#         print("NOW WERE LOOKING AT ADDITIONAL INPUT DATA", flush=True)
#         with tf.gfile.Open(FLAGS.additional_data, "r") as f:
#             for l in f:
#                 item = json.loads(l)
#                 # This little hack is because we don't want to tokenize the article twice
#                 context_ids = _flatten_and_tokenize_metadata(encoder=encoder, item=item)
#                 additional_data[item['label']].append({
#                     'info': item,
#                     'ids': context_ids,
#                     'label': item['label'],
#                 })

#     tf.logging.info("*** Done parsing files ***")
#     print("LETS GO", flush=True)
#     if FLAGS.max_training_examples > 0:

#         examples_by_label = {'human': [], 'machine': []}
#         for x in examples['train']:
#             examples_by_label[x['label']].append(x)

#         new_examples = []
#         print("Unique machine examples: {} -> {}".format(len(examples_by_label['machine']),
#                                                          FLAGS.max_training_examples), flush=True)
#         machine_ex_to_keep = examples_by_label['machine'][:FLAGS.max_training_examples]

#         # So we just cut down on the TRUE machine examples. now lets try adding in additional examples
#         # examples_by_label['human'].extend(additional_data['human'])

#         if len(additional_data['machine']) > 0:
#             amount_to_add = len(examples_by_label['human']) - len(machine_ex_to_keep)
#             if amount_to_add > 0:
#                 machine_ex_to_keep.extend(additional_data['machine'][:amount_to_add])

#         for i, human_ex in enumerate(examples_by_label['human']):
#             new_examples.append(human_ex)
#             new_examples.append(machine_ex_to_keep[i % len(machine_ex_to_keep)])

#         print("Length of examples: {} -> {}".format(len(examples['train']), len(new_examples)), flush=True)
#         examples['train'] = new_examples

#     # Training
#     if FLAGS.do_train:
#         num_train_steps = int((len(examples['train']) / FLAGS.batch_size) * FLAGS.num_train_epochs)
#         num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
#         assert num_train_steps > 0
#     else:
#         num_train_steps = None
#         num_warmup_steps = None

#     # Boilerplate
#     tpu_cluster_resolver = None
#     if FLAGS.use_tpu and FLAGS.tpu_name:
#         tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
#             FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

#     is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
#     run_config = tf.contrib.tpu.RunConfig(
#         cluster=tpu_cluster_resolver,
#         master=FLAGS.master,
#         model_dir=FLAGS.output_dir,
#         save_checkpoints_steps=FLAGS.iterations_per_loop,
#         keep_checkpoint_max=None,
#         tpu_config=tf.contrib.tpu.TPUConfig(
#             iterations_per_loop=FLAGS.iterations_per_loop,
#             num_shards=FLAGS.num_tpu_cores,
#             per_host_input_for_training=is_per_host))

#     model_fn = classification_model_fn_builder(
#         news_config,
#         init_checkpoint=FLAGS.init_checkpoint,
#         learning_rate=FLAGS.learning_rate,
#         num_train_steps=num_train_steps,
#         num_warmup_steps=num_warmup_steps,
#         use_tpu=FLAGS.use_tpu,
#         num_labels=len(LABEL_LIST),
#         pool_token_id=encoder.begin_summary,
#         adafactor=FLAGS.adafactor
#     )

#     # If TPU is not available, this will fall back to normal Estimator on CPU
#     # or GPU.
#     estimator = tf.contrib.tpu.TPUEstimator(
#         use_tpu=FLAGS.use_tpu,
#         model_fn=model_fn,
#         config=run_config,
#         train_batch_size=FLAGS.batch_size,
#         eval_batch_size=FLAGS.batch_size,
#         predict_batch_size=FLAGS.batch_size,
#         params={'model_dir': FLAGS.output_dir}
#     )

#     if FLAGS.do_train:
#         train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

#         tf.logging.info(f"***** Recreating training file at {train_file} *****")
#         classification_convert_examples_to_features(examples['train'], batch_size=FLAGS.batch_size,
#                                                     max_seq_length=FLAGS.max_seq_length,
#                                                     encoder=encoder, output_file=train_file,
#                                                     labels=LABEL_LIST,
#                                                     chop_from_front_if_needed=False)
#         tf.logging.info("***** Running training *****")
#         tf.logging.info("  Num examples = %d", len(examples['train']))
#         tf.logging.info("  Num epochs = %d", FLAGS.num_train_epochs)
#         tf.logging.info("  Batch size = %d", FLAGS.batch_size)
#         tf.logging.info("  Num steps = %d", num_train_steps)

#         train_input_fn = classification_input_fn_builder(input_file=train_file, seq_length=FLAGS.max_seq_length,
#                                                          is_training=True, drop_remainder=True,
#                                                          )
#         estimator.train(input_fn=train_input_fn, steps=num_train_steps)

#     splits_to_predict = [x for x in ['val', 'test'] if getattr(FLAGS, f'predict_{x}')]
#     for split in splits_to_predict:
#         num_actual_examples = len(examples[split])

#         predict_file = os.path.join(FLAGS.output_dir, f'{split}.tf_record')
#         tf.logging.info(f"***** Recreating {split} file {predict_file} *****")
#         classification_convert_examples_to_features(examples[split], batch_size=FLAGS.batch_size,
#                                                     max_seq_length=FLAGS.max_seq_length,
#                                                     encoder=encoder, output_file=predict_file,
#                                                     labels=LABEL_LIST, pad_extra_examples=True,
#                                                     chop_from_front_if_needed=False)

#         val_input_fn = classification_input_fn_builder(input_file=predict_file, seq_length=FLAGS.max_seq_length,
#                                                        is_training=False, drop_remainder=True,
#                                                        )

#         probs = np.zeros((num_actual_examples, 2), dtype=np.float32)
#         for i, res in enumerate(estimator.predict(input_fn=val_input_fn, yield_single_examples=True)):
#             if i < num_actual_examples:
#                 probs[i] = res['probs']

#         _save_np(os.path.join(FLAGS.output_dir, f'{split}-probs.npy'), probs)

        #preds = np.argmax(probs, 1)
        # labels = np.array([LABEL_INV_MAP[x['label']] for x in examples[split][:num_actual_examples]])
        # print('{} ACCURACY IS {:.3f}'.format(split, np.mean(labels == preds)), flush=True)


###########################################################################################################################

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from lm.dataloader import classification_convert_examples_to_features, classification_input_fn_builder
from lm.modeling import classification_model_fn_builder, GroverConfig
from lm.utils import _save_np
from sample.encoder import get_encoder
from collections import OrderedDict

flags = tf.flags

FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
    "config_file", 'configs/base.json',
    "The config json file corresponding to the pre-trained news model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_data", None,
    "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string(
    "additional_data", None,
    "Should we provide additional input data? maybe.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained model).")

# Attacking Recipe (PWWSRen2019, BAEGarg2019, BERTAttackLi2020)
flags.DEFINE_string(
    "recipe", 'PWWS',
    "used attacking recipe.")

flags.DEFINE_integer(
    "max_seq_length", 1024,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("batch_size", 1, "Batch size used")

flags.DEFINE_integer("max_training_examples", -1, "if you wanna limit the number")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("predict_val", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "predict_test", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("adafactor", False, "Whether to run adafactor")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

tf.enable_eager_execution()

from textattack.models.wrappers import ModelWrapper

# item_template = {"article": "", "domain": "", "title": "", "date": "", "authors": "", "url": "", "label": ""}


LABEL_LIST = ['machine', 'human']
LABEL_INV_MAP = {label: i for i, label in enumerate(LABEL_LIST)}

sample_item = {}

def _flatten_and_tokenize_metadata(encoder, item):
    """
    Turn the article into tokens
    :param item: Contains things that need to be tokenized
    fields are ['domain', 'date', 'authors', 'title', 'article', 'summary']
    :return: dict
    """
    metadata = []
    for key in ['domain', 'date', 'authors', 'title', 'article']:
        val = item.get(key, None)
        if val is not None:
            metadata.append(encoder.__dict__[f'begin_{key}'])
            metadata.extend(encoder.encode(val))
            metadata.append(encoder.__dict__[f'end_{key}'])
    return metadata

encoder = get_encoder()
news_config = GroverConfig.from_json_file(FLAGS.config_file)

model_fn = classification_model_fn_builder(
    news_config,
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=None,
    num_warmup_steps=None,
    use_tpu=FLAGS.use_tpu,
    num_labels=len(LABEL_LIST),
    pool_token_id=encoder.begin_summary,
    adafactor=FLAGS.adafactor
)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    cluster=None,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.iterations_per_loop,
    keep_checkpoint_max=None,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host)
    )

batch_size_large = 32
batch_size_medium = 10
batch_size_small = 1

class CustomTensorFlowModelWrapper(ModelWrapper):
    global sample_item, model_fn, run_config, news_config, batch_size_large, batch_size_medium, batch_size_small

    def __init__(self, model):
        self.model = None
        self.estimator_large = model[0]
        self.estimator_medium = model[1]
        self.estimator_small = model[2]

    def __call__(self, article_input_list):
        encoder = get_encoder()
        
        input_list = []
        for article in article_input_list:
            item = dict(sample_item)

            # item = dict_intput_list[dict_intput_list_idx[0]]
            item['article'] = article


            context_ids = _flatten_and_tokenize_metadata(encoder=encoder, item=item)
            input_list.append({
                'info': item,
                'ids': context_ids, 
                'label': item['label'],
            })
            assert item['label'] in LABEL_INV_MAP

            # dict_intput_list_idx[0] += 1

        batch_size = 1
        if len(input_list) >= batch_size_large:
            batch_size = batch_size_large
            estimator = self.estimator_large
        elif len(input_list) < batch_size_medium:
            batch_size = batch_size_small
            estimator = self.estimator_small
        else:
            batch_size = batch_size_medium
            estimator = self.estimator_medium
        

        predict_file = os.path.join(FLAGS.output_dir, 'test.tf_record')
        classification_convert_examples_to_features(input_list, batch_size=batch_size,
                                                    max_seq_length=FLAGS.max_seq_length,
                                                    encoder=encoder, output_file=predict_file,
                                                    labels=LABEL_LIST, pad_extra_examples=True,
                                                    chop_from_front_if_needed=False)

        val_input_fn = classification_input_fn_builder(input_file=predict_file, seq_length=FLAGS.max_seq_length,
                                                       is_training=False, drop_remainder=True,
                                                       )

        probs = np.zeros((len(input_list), 2), dtype=np.float32)

        for i, res in enumerate(estimator.predict(input_fn=val_input_fn, yield_single_examples=True)):
            if i < len(input_list):
                probs[i] = res['probs']

        return probs

    def get_grad(self, text_input):
        raise NotImplementedError()


from textattack.attack_recipes import PWWSRen2019, BAEGarg2019, BERTAttackLi2020

def main(_):
    global sample_item, batch_size_large, batch_size_medium, batch_size_small

    encoder = get_encoder()
    np.random.seed(123456)

    label_index = {
        "machine": 0,
        "human": 1
    }
    
    estimator_large = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=batch_size_large,
            eval_batch_size=batch_size_large,
            predict_batch_size=batch_size_large,
            params={'model_dir': FLAGS.output_dir}
        )

    estimator_medium = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=batch_size_medium,
            eval_batch_size=batch_size_medium,
            predict_batch_size=batch_size_medium,
            params={'model_dir': FLAGS.output_dir}
        )

    estimator_small = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=batch_size_small,
            eval_batch_size=batch_size_small,
            predict_batch_size=batch_size_small,
            params={'model_dir': FLAGS.output_dir}
        )

    model_wrapper = CustomTensorFlowModelWrapper([estimator_large, estimator_medium, estimator_small])

    if FLAGS.recipe == 'PWWS':
        attack = PWWSRen2019.build(model_wrapper)
    elif FLAGS.recipe == 'BAE':
        attack = BAEGarg2019.build(model_wrapper)
    elif FLAGS.recipe == 'BERTAttack':
        attack = BERTAttackLi2020.build(model_wrapper)

    with open(FLAGS.input_data, "r") as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            sample_item = dict(item)

            article = item['article']
            dataset_now = [(article, label_index[item['label']])]

            # print(model_wrapper([article]))

        # print("####################################################################")
        # print(dataset_now)
        # print("\n")
        # print("####################################################################")

            results_iterable = attack.attack_dataset(dataset_now)
            for result in results_iterable:
                print(result.__str__(color_method='ansi'))




if __name__ == "__main__":
    flags.mark_flag_as_required("input_data")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()