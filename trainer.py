from datasets import *
from metrics import *
from models import *
from tqdm import tqdm
import time


def fit(model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        schedulers,
        max_epochs,
        metrics,
        use_gpu=True,
        start_epoch=0,
        checkpoint_path=None,
        save_path=None):
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    print("Start Training ...")
    best_loss = 1e100
    best_acc = 0
    scheduler, scheduler_name = schedulers

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        # best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)

    for i in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, max_epochs):
        if scheduler_name != "plateau":
            scheduler.step()

        since = time.time()

        do_train(model, train_loader, optimizer, loss_fn, metrics, use_gpu)
        epoch_loss, metrics = do_test(model, val_loader, loss_fn, metrics, use_gpu)
        if scheduler_name == "plateau":
            scheduler.step(metrics=epoch_loss)
        acc = 0
        for metric in metrics:
            if metric.name() == 'Accuracy':
                acc = metric.value()
        save_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'loss': epoch_loss,
            'optimizer': optimizer.state_dict(),
            'accuracy': acc
        }
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(save_checkpoint, save_path + '/' + 'best-loss-%s.pth' % best_loss)
            torch.save(model, save_path + '/' + 'best-loss-model.pth')
        if acc > best_acc:
            best_acc = acc
            torch.save(save_checkpoint, save_path + '/' + 'best-acc-%s.pth' % best_acc)
            torch.save(model, save_path + '/' + 'best-acc-model.pth')
        torch.save(save_checkpoint, save_path + '/' + 'last-checkpoint.pth')

        time_elapsed = time.time() - since
        time_str = 'Total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("%s, Best loss %f, Best accuracy" % (time_str, best_loss, best_acc))

    print("Finished ...")
def do_train(model,
             train_loader,
             optimizer,
             loss_fn,
             metrics,
             use_gpu=False):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    pbar = tqdm(train_loader, desc='Training: ')
    for batch in pbar:
        inputs = batch['input']
        targets = batch['target']

        if use_gpu:
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')

        optimizer.zero_grad()
        preds, feat = model(inputs)

        loss = loss_fn(preds, feat, targets)
        losses.append(loss[0].item())
        total_loss += loss[0].item()

        loss[0].backward()
        optimizer.step()
        for metric in metrics:
            metric(preds, targets, loss)

        ordered_dict = {'lr': utils.get_lr(optimizer), 'Loss': np.mean(losses)}
        for metric in metrics:
            ordered_dict[metric.name()] = metric.value()

        pbar.set_postfix(ordered_dict)


def do_test(model,
            val_loader,
            loss_fn,
            metrics,
            use_gpu):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        model.eval()
        losses = []
        val_loss = 0
        pbar = tqdm(val_loader, desc='Validate: ')
        for batch in pbar:
            inputs = batch['input']
            targets = batch['target']

            if use_gpu:
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')

            preds, feat = model(inputs)
            loss = loss_fn(preds, feat, targets)

            losses.append(loss[0].item())
            val_loss += loss[0].item()

            for metric in metrics:
                metric(preds, targets, loss)

            ordered_dict = {'Loss': np.mean(losses)}
            for metric in metrics:
                ordered_dict[metric.name()] = metric.value()

            pbar.set_postfix(ordered_dict)
        return np.mean(losses), metrics


if __name__ == '__main__':
    pass
