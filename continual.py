import torch
import argparse
import bayesfunc as bf

from torch import optim
from pathlib import Path
from data import get_data
from logs import get_log_directory
from nets import get_net, get_old_net
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Globally Induced Variational Continual Learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--type", default="global", type=str, help="Network to train")
parser.add_argument("--tasks", default=10, type=int, help="Tasks to train on")
parser.add_argument("--epochs", default=800, type=int, help="Epochs to train for")
parser.add_argument("--learning-rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--train-samples", default=3, type=int, help="Train samples")
parser.add_argument("--test-samples", default=10, type=int, help="Test samples")
parser.add_argument("--batch-size", default=1024, type=int, help="Batch size")
parser.add_argument("--inducing-size", default=1000, type=int, help="Inducing size")
parser.add_argument("--log-frequency", default=1000, type=int, help="Logging frequency in steps")
parser.add_argument("--val-frequency", default=80, type=int, help="Validation frequency in epochs")

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    log_directory = get_log_directory(args)
    summary_writer = SummaryWriter(str(log_directory), flush_secs=5)

    train_loaders, test_loaders, data_size = get_data(args.tasks, args.batch_size)
    net = get_net(args.type, args.inducing_size).to(device)

    train(net, args.tasks, args.epochs, args.learning_rate, args.train_samples, args.test_samples, train_loaders, test_loaders, data_size, args.log_frequency, args.val_frequency, summary_writer)
    summary_writer.close()

def train(net, tasks, epochs, learning_rate, train_samples, test_samples, train_loaders, test_loaders, data_size, log_frequency, val_frequency, summary_writer):
    step = 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for task in range(tasks):
        old_net = get_old_net(net).to(device)
        for epoch in range(epochs):
            net.train()
            for batch, targets in train_loaders[task]:
                batch, targets = batch.to(device), targets.to(device)
                logits, new_logpq, new_weights = bf.propagate(net, batch.expand(train_samples, -1, -1))
                _, old_logpq, _ = bf.propagate(old_net, batch.expand(train_samples, -1, -1), sample_dict=new_weights, detach=False) 
                logpq = new_logpq if (task == 0) else new_logpq - old_logpq

                probs = Categorical(logits=logits)
                logloss = probs.log_prob(targets).mean(1)
                
                elbo = -(logloss.mean() + logpq.mean() / data_size[task])
                elbo.backward()

                optimizer.step()
                optimizer.zero_grad()

                if ((step + 1) % log_frequency == 0):
                    summary_writer.add_scalar("Elbo", elbo.item(), step)
                step += 1

            if ((epoch + 1) % val_frequency == 0 or (epoch + 1) == epochs):
                test(net, task, epoch, step, test_samples, test_loaders, summary_writer, (epoch + 1) == epochs)
        
def test(net, tasks, epochs, steps, test_samples, test_loaders, summary_writer, final_test):
    total_logloss = []
    total_accuracy = []
    with torch.no_grad():
        for task in range(tasks + 1):
            net.eval()
            for batch, targets in test_loaders[task]:
                batch, targets = batch.to(device), targets.to(device)
                logits, _, _ = bf.propagate(net, batch.expand(test_samples, -1, -1))
                logits = logits.log_softmax(-1).logsumexp(0)

                probs = Categorical(logits=logits)
                logloss = -(probs.log_prob(targets).mean(0))

                preds = logits.argmax(1)
                accuracy = (preds == targets).float().mean()

                total_logloss.append(logloss.item())
                total_accuracy.append(accuracy.item())
        
        average_logloss = torch.tensor(total_logloss).mean().item()
        average_accuracy = torch.tensor(total_accuracy).mean().item()

        summary_writer.add_scalar("Loss", average_logloss, steps)
        summary_writer.add_scalar("Accuracy", average_accuracy, steps)

        if (final_test == True): 
            with open(Path(str(summary_writer.log_dir).replace("logs", "outputs") + ".csv"), 'a') as file: 
                file.write(str(average_accuracy) + "\n")

        print(f"Task: [{tasks + 1}] Epoch: [{epochs + 1}] Loss: {average_logloss:.5f} Accuracy: {average_accuracy:.5f}")

if __name__ == "__main__":
    main(parser.parse_args())