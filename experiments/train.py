from fen.model.fen import FEN
from fen.dataset import CGPDataset
from fen.trainer import Trainer

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
num_epochs = 40

def main():
    train_dataset = CGPDataset("datasets/ariths_gen_100_high_prec_augmented/train")
    val_dataset = CGPDataset("datasets/ariths_gen_100_high_prec_augmented/val")


    print('[INFO] Create Model and Trainer')
    model = FEN()
    model.load("./pretrained/ariths_gen1.pth")
    print(f'[INFO] Model created parameters: {model.n_param()}')

    trainer = Trainer(model, training_id="small", lr=5e-3)

    print('[INFO] Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)


if __name__ == '__main__':
    main()
