import time
from tqdm import tqdm
import os, re, argparse
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.utils import compute_class_weight
import numpy as np
import pandas as pd
from my_util import *
from DeepLineDP_model import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(0)

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-batch_size', type=int, default=32)
arg.add_argument('-num_epochs', type=int, default=30)
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size')
arg.add_argument('-word_gcn_hidden_dim', type=int, default=128, help='word attention hidden size')
arg.add_argument('-sent_gcn_hidden_dim', type=int, default=128, help='sentence attention hidden size')
arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')
arg.add_argument('-lr', type=float, default=0.001, help='learning rate')
arg.add_argument('-exp_name',type=str,default='')
arg.add_argument('-weighted_graph', type=bool, default=False, help='Whether to use weighted graph')

args = arg.parse_args()

# model setting
batch_size = args.batch_size
num_epochs = args.num_epochs
max_grad_norm = 5
embed_dim = args.embed_dim
word_gcn_hidden_dim = args.word_gcn_hidden_dim
sent_gcn_hidden_dim = args.sent_gcn_hidden_dim
weighted_graph = args.weighted_graph
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = args.dropout
lr = args.lr
save_every_epochs = 1
exp_name = args.exp_name

max_train_LOC = 900

prediction_dir = '../output/prediction/LineDef/'
save_model_dir = '../output/model/LineDef/'

file_lvl_gt = '../../datasets/preprocessed_data/'

weight_dict = {}


def get_loss_weight(labels):
    '''
        input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1, 1).to(device)
    return weight_tensor

def train_model(dataset_name):
    start_time = time.time()
    loss_dir = '../output/loss/LineDef/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)


    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    train_df = get_df(train_rel)
    valid_df = get_df(valid_rel)

    train_code3d, train_label, train_word_edge, tarin_line_edge = get_code3d_and_label(train_df, True)
    valid_code3d, valid_label, valid_word_edge, valid_line_edge = get_code3d_and_label(valid_df, True)

    sample_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)

    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)

    w2v_dir = get_w2v_path()

    word2vec_file_dir = os.path.join(w2v_dir, dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for', dataset_name, 'finished')

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    vocab_size = len(word2vec.wv.vocab) + 1  # for unknown tokens

    x_train_vec = get_x_vec(train_code3d, word2vec)

    x_valid_vec = get_x_vec(valid_code3d, word2vec)

    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)

    train_label_tensor = torch.FloatTensor([label for label in train_label])
    tarin_code_vec_pad = torch.tensor(pad_code(x_train_vec, max_sent_len))
    train_padded_edge = pad_word_edge_index(train_word_edge, max_sent_len)
    tarin_line_edge = pad_line_edge_index(tarin_line_edge, max_sent_len)

    valid_label_tensor = torch.FloatTensor([label for label in valid_label])
    valid_code_vec_pad = torch.tensor(pad_code(x_valid_vec, max_sent_len))
    valid_padded_edge = pad_word_edge_index(valid_word_edge, max_sent_len)
    valid_line_edge = pad_line_edge_index(valid_line_edge, max_sent_len)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("get all data spend {} s".format(elapsed_time))

    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gcn_hidden_dim=word_gcn_hidden_dim,
        sent_gcn_hidden_dim=sent_gcn_hidden_dim,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        device=device)

    model = model.to(device)

    model.sent_attention.word_attention.freeze_embeddings(False)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion = nn.BCELoss()

    checkpoint_files = os.listdir(actual_save_model_dir)

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')

    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:
        model.sent_attention.word_attention.init_embeddings(word2vec_weights.to(device))
        current_checkpoint_num = 1

        train_loss_all_epochs = []
        val_loss_all_epochs = []
    
    else:
        checkpoint_nums = [int(re.findall('\d+', s)[0]) for s in checkpoint_files]
        current_checkpoint_num = max(checkpoint_nums)

        checkpoint = torch.load(actual_save_model_dir+'checkpoint_' + exp_name + '_'+str(current_checkpoint_num)+'epochs.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        loss_df = pd.read_csv(loss_dir+dataset_name+'-loss_record.csv')
        train_loss_all_epochs = list(loss_df['train_loss'])
        val_loss_all_epochs = list(loss_df['valid_loss'])

        current_checkpoint_num = current_checkpoint_num+1  # go to next epoch
        print('continue training model from epoch', current_checkpoint_num)
    columns = ['Epoch', 'Train Loss', 'Train Balanced Accuracy', 'Validation Loss', 'Validation Balanced Accuracy']
    df_results = pd.DataFrame(columns=columns)
    for epoch in tqdm(range(current_checkpoint_num, num_epochs+1)):
        train_losses = []
        val_losses = []
        train_predictions = []
        val_predictions = []
        train_labels_all = []
        val_labels_all = []

        model.train()

        for inputs, labels, word_edge_index, line_edge_index in batch_generator(tarin_code_vec_pad, train_label_tensor, train_padded_edge, tarin_line_edge, batch_size):

            output, _, __, ___ = model(inputs, word_edge_index, line_edge_index)

            labels = labels.to(device)

            weight_tensor = get_loss_weight(labels)

            criterion.weight = weight_tensor

            loss = criterion(output, labels.reshape(batch_size,1))

            train_losses.append(loss.item())

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            predictions = (output >= 0.5).int().cpu().numpy().reshape(-1)
            # print("predictions: ", predictions)
            train_predictions.extend(predictions)
            train_labels_all.extend(labels.cpu().numpy().reshape(-1))
            # print("train_labels_all: ", train_labels_all)
        #
        train_balanced_acc = balanced_accuracy_score(train_labels_all, train_predictions)
        # print("train_balanced_acc: ", train_balanced_acc)
        train_loss_all_epochs.append(np.mean(train_losses))

        with torch.no_grad():
            
            criterion.weight = None
            model.eval()

            for inputs, labels, word_edge_index, line_edge_index in batch_generator(valid_code_vec_pad, valid_label_tensor, valid_padded_edge, valid_line_edge, batch_size):

                output, _, __, ___ = model(inputs, word_edge_index, line_edge_index)

                labels = labels.to(device)

                val_loss = criterion(output, labels.reshape(batch_size,1))

                val_losses.append(val_loss.item())
                predictions = (output >= 0.5).int().cpu().numpy().reshape(-1)
                val_predictions.extend(predictions)
                val_labels_all.extend(labels.cpu().numpy().reshape(-1))

            val_balanced_acc = balanced_accuracy_score(val_labels_all, val_predictions)
            val_loss_all_epochs.append(np.mean(val_losses))

        if epoch % save_every_epochs == 0:
            # print(dataset_name,'- at epoch:',str(epoch))

            if exp_name == '':
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+str(epoch)+'epochs.pth')
            else:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+exp_name+'_'+str(epoch)+'epochs.pth')

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(loss_dir+dataset_name+'-loss_record.csv',index=False)
        df_results = df_results.append({
            'Epoch': epoch,
            'Train Loss': np.nanmean(train_losses),
            'Train Balanced Accuracy': train_balanced_acc,
            'Validation Loss': np.nanmean(val_losses),
            'Validation Balanced Accuracy': val_balanced_acc
        }, ignore_index=True)

    df_results.to_csv(f'../output/{exp_name}_{dataset_name}_loss.csv', index=False)

dataset_name = args.dataset
train_model(dataset_name)