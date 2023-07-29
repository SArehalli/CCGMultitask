import torch
import torch.nn as nn

import numpy as np

def evaluate_lm(model, loader, batch_size, loss_f, cuda=True, batch_first=False, wrap=lambda x: x):
    model.eval()
    hidden = model.init_hidden(batch_size)
    loss_lm = 0.0

    with torch.no_grad():
        for (input, target_lm) in wrap(loader):
            # Move to GPU if necessary
            if cuda:
                input = input.cuda()
                target_lm = target_lm.cuda()

            # data comes as batch x seq_len, lstm takes seq_len x batch
            input = input.transpose(0,1).contiguous()
            target_lm = target_lm.transpose(0,1).contiguous()
            
            # Do an forward pass
            out = model(input, hidden)
            out_lm = out[0]
            hidden = out[-1] # works regardless of # of args
            # change LSTM output from seq_len x batch x word_idx to (seq_len * batch) x word_idx
            # change target from seq_len x batch to (seq_len * batch)
            # compute loss (softmax is in the extractor)
            loss_lm += loss_f(out_lm, target_lm.view(-1)).item()

    return loss_lm/len(loader)

def get_topk_acc(target, output, k):
    # target_ccg is seq_len x batch -> (seq_len * batch)
    target = target.view(-1)
    topk_idxs = torch.topk(output, k, dim=1)[1]
    assert(len(target) == len(topk_idxs))

    # Count of k-best matches in batch
    one_best_count = 0
    for i, gold_tag in enumerate(target):
        if gold_tag.item() in topk_idxs[i]:
            one_best_count += 1

    # count of tokens in batch
    seq_len = target.view(-1).size()[0]

    return one_best_count/seq_len

def evaluate_ccg(model, loader, batch_size, loss_f, cuda=True, batch_first=False, wrap=lambda x: x, nbest=1):
    model.eval()
    hidden = model.init_hidden(batch_size)
    loss_ccg = 0.0
    one_best_acc = 0.0

    with torch.no_grad():
        for (input, target_ccg) in wrap(loader):
            # Move to GPU if necessary
            if cuda:
                input = input.cuda()
                target_ccg = target_ccg.cuda()

            # data comes as batch x seq_len, lstm takes seq_len x batch
            input = input.transpose(0,1).contiguous()
            target_ccg = target_ccg.transpose(0,1).contiguous()
            
            # Do an forward pass
            _, out_ccg, hidden = model(input, hidden)

            # within batch accuracy
            one_best_acc += get_topk_acc(target_ccg, out_ccg, nbest)

            # change LSTM output from seq_len x batch x tag_idx to (seq_len * batch) x tag_idx
            # change target from seq_len x batch to (seq_len * batch)
            # compute loss (softmax is in the extractor)
            loss_ccg += loss_f(out_ccg, target_ccg.view(-1)).item()

    return loss_ccg/len(loader), one_best_acc/len(loader)

def train_augment(model, optimizer, weight_lm,
                  train_loader, valid_loader_lm, 
                  valid_loader_ccg, batch_size,
                  loss_f, clip, log_interval, 
                  max_epochs, save_path, cuda=True, batch_first=False, 
                  early_stop=False, wrap=lambda x: x, patience=3,
                  init_epoch=0, save_all = False):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.25, verbose=True)
    best_loss = np.inf
    losses = []

    def wsum(pair):
        return (weight_lm * pair[0] + (1.0 - weight_lm) * pair[1])

    for epoch in wrap(range(max_epochs)):
        total_loss = {"lm":0.0, "ccg":0.0}
        prev_total_loss = {"lm":0.0, "ccg":0.0}
        hidden_lm = model.init_hidden(batch_size)
        hidden_ccg = model.init_hidden(batch_size)
        for i, ((input_lm, target_lm), (input_ccg, target_ccg)) in enumerate(train_loader):
            model.train()
            loss_lm, loss_ccg = 0, 0
            if weight_lm > 0.0:
                # Move to GPU if needed
                if cuda:
                    input_lm = input_lm.cuda()
                    target_lm = target_lm.cuda()

                # Targets are batch x seq_len, so transpose to seq_len x batch
                # (to match RNN format of seq_len x batch x word/tag_idx)
                input_lm = input_lm.transpose(0,1).contiguous()
                target_lm = target_lm.transpose(0,1).contiguous()

                ## Detach to enforce bptt limit
                hidden_lm = (hidden_lm[0].detach(), hidden_lm[1].detach())
                
                out_lm, _, hidden_lm = model(input_lm, hidden_lm)

                # LM loss
                ## out from seq_len x batch x word_idx -> (seq_len * batch) x word_idx
                ## target from seq_len x batch -> (seq_len * batch)
                loss_lm = loss_f(out_lm, target_lm.view(-1))
                
                # Sum the total loss for this epoch
                total_loss["lm"] += loss_lm.item()

            if weight_lm < 1.0:
                model.train()

                # Move to GPU if needed
                if cuda:
                    input_ccg = input_ccg.cuda()
                    target_ccg = target_ccg.cuda()

                # Targets are batch x seq_len, so transpose to seq_len x batch
                # (to match RNN format of seq_len x batch x word/tag_idx)
                input_ccg = input_ccg.transpose(0,1).contiguous()
                target_ccg = target_ccg.transpose(0,1).contiguous()

                ## Detach to enforce bptt limit
                hidden_ccg = (hidden_ccg[0].detach(), hidden_ccg[1].detach())
                
                _, out_ccg, hidden_ccg = model(input_ccg, hidden_ccg)

                # CCG loss

                ## out from seq_len x batch x word_idx -> (seq_len * batch) x word_idx
                ## target from seq_len x batch -> (seq_len * batch)
                loss_ccg = loss_f(out_ccg, target_ccg.view(-1))
                    
                # Sum the total loss for this epoch
                total_loss["ccg"] += loss_ccg.item()

            # weight and sum the loss
            loss = (weight_lm * loss_lm) + ((1.0 - weight_lm) * loss_ccg)

            # Compute gradients from the weighted loss
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()

            # Clip large gradients
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # adjust weights
            optimizer.step()


        # Compute validation loss at each interval (a triple of lm loss, ccg loss, and ccg acc)
        valid_loss_lm = evaluate_lm(model, valid_loader_lm, batch_size, loss_f)
        valid_loss_ccg = evaluate_ccg(model, valid_loader_ccg, batch_size, loss_f)
        valid_loss = [valid_loss_lm] + list(valid_loss_ccg)

        # Average the training loss 
        train_loss = wsum((total_loss["lm"], total_loss["ccg"]))/(len(train_loader))
        # keep list of train/valid losses to return (for plotting)
        losses.append((train_loss, wsum(valid_loss)))

        # Save and record if we've gotten a new best model
        best = " "
        if (wsum(valid_loss) < best_loss):
            best_loss = wsum(valid_loss)

            torch.save(model.state_dict(), save_path + "_best.pt")
            optim_state = {"optimizer":optimizer.state_dict(),
                           "epoch":epoch}
            torch.save(optim_state, save_path + ".opt")
            best = "*"
        elif save_all:
            torch.save(model.state_dict(), save_path + "ep{}.pt".format(init_epoch + epoch))
            optim_state = {"optimizer":optimizer.state_dict(),
                           "epoch":epoch}
            torch.save(optim_state, save_path + ".opt")
            

        print(("epoch {:2}{} \t| batch {:2} \t| train lm {:.5f} ccg {:.5f} total {:.5f} "
               "\n\t\t|  valid lm nll {:.5f} ppl {:6.2f} \t| valid ccg nll {:.5f} ppl {:6.2f} 1-best {:.5f}").format(
              init_epoch + epoch, best, i, total_loss["lm"]/len(train_loader), total_loss["ccg"]/len(train_loader), train_loss, 
              valid_loss[0], np.exp(valid_loss[0]), valid_loss[1], np.exp(valid_loss[1]), valid_loss[2]))


        # step the scheduler to see if we reduce the LR
        scheduler.step(wsum(valid_loss))


def train_lm(model, optimizer, 
             train_loader, valid_loader_lm, 
             batch_size,
             loss_f, clip, log_interval, 
             max_epochs, save_path, cuda=True, batch_first=False, 
             early_stop=False, wrap=lambda x: x, patience=3,
             init_epoch=0, save_all=False):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.25, verbose=True)
    best_loss = np.inf
    losses = []


    for epoch in wrap(range(max_epochs)):
        total_loss = {"lm":0.0}
        prev_total_loss = {"lm":0.0}
        hidden_lm = model.init_hidden(batch_size)
        for i, (input_lm, target_lm) in enumerate(train_loader):
            model.train()
            loss_lm = 0, 0
            # Move to GPU if needed
            if cuda:
                input_lm = input_lm.cuda()
                target_lm = target_lm.cuda()

            # Targets are batch x seq_len, so transpose to seq_len x batch
            # (to match RNN format of seq_len x batch x word/tag_idx)
            input_lm = input_lm.transpose(0,1).contiguous()
            target_lm = target_lm.transpose(0,1).contiguous()

            ## Detach to enforce bptt limit
            hidden_lm = (hidden_lm[0].detach(), hidden_lm[1].detach())
            
            out_lm, hidden_lm = model(input_lm, hidden_lm)

            # LM loss
            ## out from seq_len x batch x word_idx -> (seq_len * batch) x word_idx
            ## target from seq_len x batch -> (seq_len * batch)
            loss = loss_f(out_lm, target_lm.view(-1))
            
            # Sum the total loss for this epoch
            total_loss["lm"] += loss.item()

            # Compute gradients from the weighted loss
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()

            # Clip large gradients
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # adjust weights
            optimizer.step()


        # Compute validation loss at each interval (a triple of lm loss, ccg loss, and ccg acc)
        valid_loss = evaluate_lm(model, valid_loader_lm, batch_size, loss_f)

        # Average the training loss 
        train_loss = (total_loss["lm"]/(len(train_loader)))
        # keep list of train/valid losses to return (for plotting)
        losses.append((train_loss, valid_loss))

        # Save and record if we've gotten a new best model
        best = " "
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save_path + ".pt")
            optim_state = {"optimizer":optimizer.state_dict(),
                           "epoch":epoch}
            torch.save(optim_state, save_path + ".opt")
            best = "*"
        elif save_all:
            torch.save(model.state_dict(), save_path + "ep{}.pt".format(init_epoch + epoch))
            optim_state = {"optimizer":optimizer.state_dict(),
                           "epoch":epoch}
            torch.save(optim_state, save_path + ".opt")

        print(("epoch {:2}{} \t| batch {:2} \t| train lm {:.5f}"
               "\n\t\t|  valid lm nll {:.5f} ppl {:6.2f}").format(
              init_epoch + epoch, best, i, total_loss["lm"]/len(train_loader), train_loss, 
              valid_loss, np.exp(valid_loss)))


        # step the scheduler to see if we reduce the LR
        scheduler.step(valid_loss)



if __name__ == "__main__":
   x = torch.tensor(range(50)).view(5,2,5)
   y = torch.argmax(x, dim=2)
   print(get_topk_acc(y.view(-1), x.view(-1, 5), 3))
