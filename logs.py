from pathlib import Path

def get_log_directory(args):
    if (args.type == "global"):
        log_directory_suffix = (f"{args.type}_ta={args.tasks}_ep={args.epochs}_lr={args.learning_rate}_ts={args.train_samples}_ts={args.test_samples}_bs={args.batch_size}_is={args.inducing_size}_")
    elif (args.type == "factorised"):
        log_directory_suffix = (f"{args.type}_ta={args.tasks}_ep={args.epochs}_lr={args.learning_rate}_ts={args.train_samples}_ts={args.test_samples}_bs={args.batch_size}_")
    
    i = 0
    log_directory = Path("logs") / (log_directory_suffix + str(i))
    while log_directory.exists():
        i += 1
        log_directory = Path("logs") / (log_directory_suffix + str(i))
    print(f"Writing logs to {log_directory}")
    return str(log_directory)