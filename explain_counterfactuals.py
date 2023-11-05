import argparse
import os

import numpy as np
import torch
import yaml

from explainer.counterfactuals import compute_counterfactual
from explainer.eval import compute_eval_metrics
from explainer.utils import get_query_distractor_pairs, process_dataset
from tqdm import tqdm
from explainer.common_config import (
    get_model,
    get_test_dataloader,
    get_customized_dataset,
)
from explainer.path import Path
import torch.distributed as dist
import torch.nn.parallel

parser = argparse.ArgumentParser(description="Generate counterfactual explanations")
parser.add_argument("--config_path", type=str, required=True)


def main(pretrained_model_index=None):
    args = parser.parse_args()

    # parse args
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    pretrained_model_index = config["pretrained_model_index"] if pretrained_model_index is None else pretrained_model_index

    experiment_name = os.path.basename(args.config_path).split(".")[0]
    dirpath = os.path.join(Path.output_root_dir(), experiment_name)
    os.makedirs(dirpath, exist_ok=True)

    # device
    assert torch.cuda.is_available()
    config['local_rank'] = int(os.environ["LOCAL_RANK"]) if config['use_ddp'] and os.environ.get("LOCAL_RANK") else 0
    device = torch.device(config['local_rank'])
    if config['use_ddp']:
        torch.cuda.set_device(config['local_rank'])
        torch.distributed.init_process_group(backend='nccl')

    dataset = get_customized_dataset(config['data_path'])
    dataloader, train_sampler = get_test_dataloader(config, dataset)

    # load classifier
    print("Load classification model weights")
    model = get_model(config)
    model_path = config["counterfactuals_kwargs"]["model_dir"] + 'pretrained_{}.pth'.format(pretrained_model_index)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # process dataset
    print("Pre-compute classifier predictions")
    result = process_dataset(model, dataloader, device, config)
    features = result["features"]
    preds = result["preds"].numpy()
    targets = result["targets"].numpy()
    print("Top-1 accuracy: {:.2f}".format(100 * result["top1"]))

    # compute query-distractor pairs
    print("Pre-compute query-distractor pairs")
    query_distractor_pairs = get_query_distractor_pairs(
        dataset,
        confusion_matrix=result["confusion_matrix"],
        max_num_distractors=config["counterfactuals_kwargs"][
            "max_num_distractors"
        ],  # noqa
    )

    # get classifier head
    classifier_head = model.get_classifier_head()
    classifier_head = classifier_head.to(device)
    classifier_head.eval()

    # compute counterfactuals
    print("Compute counterfactuals")
    counterfactuals = {}

    for query_index in tqdm(range(len(dataset))):
        if query_index not in query_distractor_pairs.keys():
            continue  # skips images that were classified incorrectly

        # gather query features
        query = features[query_index]  # dim x n_row x n_row
        query_pred = preds[query_index]
        if query_pred != targets[query_index]:
            continue  # skip if query classified incorrect

        # gather distractor features
        distractor_target = query_distractor_pairs[query_index][
            "distractor_class"
        ]  # noqa
        distractor_index = query_distractor_pairs[query_index][
            "distractor_index"
        ]  # noqa
        if isinstance(distractor_index, int):
            if preds[distractor_index] != distractor_target:
                continue  # skip if distractor classified is incorrect
            distractor_index = [distractor_index]

        else:  # list
            distractor_index = [
                jj for jj in distractor_index if preds[jj] == distractor_target
            ]
            if len(distractor_index) == 0:
                continue  # skip if no distractors classified correct

        distractor = torch.stack([features[jj] for jj in distractor_index], dim=0)

        query_aux_features = None
        distractor_aux_features = None

        try:
            list_of_edits = compute_counterfactual(
                query=query,
                distractor=distractor,
                classification_head=classifier_head,
                distractor_class=distractor_target,
                query_aux_features=query_aux_features,
                distractor_aux_features=distractor_aux_features,
                lambd=config["counterfactuals_kwargs"]["lambd"],
                temperature=config["counterfactuals_kwargs"]["temperature"],
                topk=config["counterfactuals_kwargs"]["topk"]
                if "topk" in config["counterfactuals_kwargs"].keys()
                else None,
                device=device,
            )

        except BaseException:
            print("warning - no counterfactual @ index {}".format(query_index))
            continue

        counterfactuals[query_index] = {
            "query_index": query_index,
            "distractor_index": distractor_index,
            "query_target": query_pred,
            "distractor_target": distractor_target,
            "edits": list_of_edits,
        }

    # save result
    np.save(os.path.join(dirpath, "counterfactuals_with_pretrained_{}.npy".format(pretrained_model_index)), counterfactuals)

    # evaluation
    print("Generated {} counterfactual explanations".format(len(counterfactuals)))
    average_num_edits = np.mean([len(res["edits"]) for res in counterfactuals.values()])
    print("Average number of edits is {:.2f}".format(average_num_edits))

    result = compute_eval_metrics(
        counterfactuals,
        dataset=dataset,
    )

    print("Eval results single edit: {}".format(result["single_edit"]))
    print("Eval results all edits: {}".format(result["all_edit"]))

    np.save(os.path.join(dirpath, "eval_results_with_pretrained_{}.npy".format(pretrained_model_index)), result)


if __name__ == "__main__":
    main()
