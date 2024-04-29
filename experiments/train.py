from fen.model.fen import FEN
from fen.dataset import CGPDataset
from fen.trainer import Trainer

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
num_epochs = 20

def main():
    train_dataset = CGPDataset("datasets/ariths_gen_test/train")
    val_dataset = CGPDataset("datasets/ariths_gen_test/val")


    print('[INFO] Create Model and Trainer')
    model = FEN()
    #model.load("./exp/size_200_2/model_7.pth")
    print(f'[INFO] Model created parameters: {model.n_param()}')

    trainer = Trainer(model, training_id="size_200_3", lr=5e-6)
    print('[INFO] Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)


if __name__ == '__main__':
    main()
