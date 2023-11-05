import torch
from utils import load_data_and_get_dataloader, EarlyStopping
import argparse
import yaml
from model import ResNet18
from tqdm import tqdm
import os
from preprocessing import preprocessing


def _trainer(config, num_classes, train_data_path, valid_data_path, save_path):

    model = ResNet18(num_classes=num_classes).to(config['local_rank'])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    train_dataloader, train_sampler = load_data_and_get_dataloader(train_data_path, config, mixup=True)
    valid_dataloader, valid_sampler = load_data_and_get_dataloader(valid_data_path, config)

    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=30, verbose=False, path=save_path)

    mixup = True
    for epoch in tqdm(range(config['n_epochs']), desc="Epochs", unit="epoch"):
        if epoch == 20:
            mixup = False

        train_sampler.set_epoch(epoch) if config['use_ddp'] else None
        # Uncomment to monitor training
        # batch_loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Batches", unit="batch",
        #                   leave=False) if config['local_rank'] == 0 or not config['use_ddp'] else train_dataloader
        batch_loop = train_dataloader

        model.train()
        for batch, (data, label) in enumerate(batch_loop):
            if not mixup:
                data, label = data.to(config['local_rank']), label[0].to(config['local_rank'])
            else:
                data, label1, label2, lam = data.to(config['local_rank']), label[0].to(config['local_rank']), \
                                            label[1].to(config['local_rank']), label[2].to(config['local_rank'])
                label = lam * label1 + (1 - lam) * label2
            optimizer.zero_grad()
            out = model(data)['logits']

            if not mixup:
                loss = torch.nn.CrossEntropyLoss()(out, label)
            else:
                loss = (lam * torch.nn.CrossEntropyLoss()(out, label1)
                        + (1 - lam) * torch.nn.CrossEntropyLoss()(out, label2)).mean()

            loss.backward()
            optimizer.step()

            # # Uncomment to monitor training
            # metrics = {'Loss': {'Cls': loss.item()},
            #            'Accuracy': {'Cls': (out.argmax(dim=1) == label).float().mean().item()}}
            # if config['local_rank'] == 0 or not config['use_ddp']:
            #     batch_loop.set_postfix(**{'Loss_' + k: "{:.3f}".format(v)[:5] for k, v in metrics['Loss'].items()},
            #                            **{'Accuracy_' + k: "{:.3f}".format(v)[:5] for k, v in
            #                               metrics['Accuracy'].items()})

        loss_valid = 0
        model.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(valid_dataloader):
                data, label = data.to(config['local_rank']), label.to(config['local_rank'])
                out = model(data)['logits']
                loss_valid += torch.nn.CrossEntropyLoss()(out, label)
            loss_valid /= len(valid_dataloader)
            early_stopping(loss_valid, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break


def trainer():
    print("Start training... It will take some time. Please wait.")
    parser = argparse.ArgumentParser(description="Generate counterfactual explanations")
    parser.add_argument("--config_path", type=str, default='configs/pretrain.yaml')
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    config['local_rank'] = int(os.environ["LOCAL_RANK"]) if config['use_ddp'] and os.environ.get("LOCAL_RANK") else 0
    class_configs = ['classes_18', 'classes_2']
    counter = 0
    for class_config in class_configs:
        counter += 1
        _trainer(
            config,
            config[class_config]['num_classes'],
            config[class_config]['train_data_dir'],
            config[class_config]['valid_data_dir'],
            config[class_config]['save_dir']
        )
        print(f"{100 * counter / 2:.2f}%")
    print("Training finished! Pretrained models saved.")


if __name__ == '__main__':
    files_to_check = [
        'BCICIV_2a_data_LR_SAMELABELs_Test.npz',
        'BCICIV_2a_data_LR_SAMELABELs_Train.npz',
        'BCICIV_2a_data_LR_SAMELABELs_Validation.npz',
        'BCICIV_2a_data_LR_UNIQUELABELs_Test.npz',
        'BCICIV_2a_data_LR_UNIQUELABELs_Train.npz',
        'BCICIV_2a_data_LR_UNIQUELABELs_Validation.npz',
    ]

    if any(not os.path.exists(f'data/{file}') for file in files_to_check):
        preprocessing()

    trainer()
