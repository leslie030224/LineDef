import re
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"

max_seq_len = 50

all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6',
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0',
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'],
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'],
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                     'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                     'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
                     'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                     'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
                     'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                     'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                     'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'

word2vec_dir = '../output/Word2Vec_model/' 


def batch_generator(code, label, word_edge, line_edge, batch_size, random_seed=0):
    total_samples = len(code)
    num_batches = total_samples // batch_size

    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)

        batch_indices = indices[start_idx:end_idx]
        batch_code = code[batch_indices]
        batch_label = label[batch_indices]
        batch_word_edge = [word_edge[idx] for idx in batch_indices]
        batch_line_edge = [line_edge[idx] for idx in batch_indices]

        yield batch_code, batch_label, batch_word_edge, batch_line_edge


def pad_word_edge_index(line_egde_index_list, max_sent_len, limit_sent_len=True):
    padded_edge_list = []
    for edge in line_egde_index_list:
        num = max_sent_len - len(edge)
        if max_sent_len - len(edge) > 0 :
            for _ in range(num):
                edge.append(np.full((2, 1), -1))

        if limit_sent_len:
            padded_edge_list.append(edge)
        else:
            padded_edge_list.append(edge)

    return padded_edge_list


def pad_line_edge_index(line_egde_index_list, max_sent_len, limit_sent_len=True):
    padded_line_edge = []
    if limit_sent_len:
        for edge_index in line_egde_index_list:
            idx = np.any(edge_index > max_sent_len - 1, axis=0)
            column_numbers = np.where(idx)[0]
            edge_index = np.delete(edge_index, column_numbers, axis=1)
            padded_line_edge.append(edge_index)
    return padded_line_edge


def get_df(rel, is_baseline=False):

    if is_baseline:
        df = pd.read_csv('../'+file_lvl_gt+rel+".csv")

    else:
        df = pd.read_csv(file_lvl_gt+rel+".csv")

    df = df.fillna('')

    df = df[df['is_blank']==False]
    df = df[df['is_test_file']==False]

    return df


def prepare_line_adj(line_lenth, weighted_graph = False):

    window_size = 6
    windows = []
    idx = range(line_lenth)

    if line_lenth <= window_size:
        windows.append(idx)
    else:
        for j in range(line_lenth - window_size + 1):
            window = idx[j: j + window_size]
            windows.append(window)

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p_id = window[p]
                word_q_id = window[q]
                # if word_p_id == word_q_id:
                #     continue
                word_pair_key = (word_p_id, word_q_id)

                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # two orders
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.

    row = []
    col = []
    weight = []
    for key in word_pair_count:
        p = key[0]
        q = key[1]
        row.append(p)
        col.append(q)
        weight.append(word_pair_count[key] if weighted_graph else 1.)

    adj = sp.csr_matrix((weight, (row, col)), shape=(line_lenth, line_lenth))
    edge_index = np.array(adj.nonzero())

    return edge_index


def prepare_code2d(code_list, line_lenth, to_lowercase=False, weighted_graph=False):

    window_size = 2
    code2d = []
    all_word_edge_index = []

    for c in code_list:
        windows = []

        c = re.sub('\\s+',' ',c)

        if to_lowercase:
            c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)

        if total_tokens > max_seq_len:
            token_list = token_list[:max_seq_len]
            total_tokens = max_seq_len

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>']*(max_seq_len-total_tokens)

        code2d.append(token_list)

        idx = range(0, total_tokens)

        if total_tokens <= window_size:
            windows.append(idx)
        else:
            for j in range(total_tokens - window_size + 1):
                window = idx[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p_id = window[p]
                    word_q_id = window[q]

                    # if word_p_id == word_q_id:
                    #     continue
                    word_pair_key = (word_p_id, word_q_id)

                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # two orders
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(p)
            col.append(q)
            weight.append(word_pair_count[key] if weighted_graph else 1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(max_seq_len, max_seq_len))
        edge_index = np.array(adj.nonzero())
        all_word_edge_index.append(edge_index)

    all_line_edge_index = prepare_line_adj(line_lenth, weighted_graph)

    return code2d, all_word_edge_index, all_line_edge_index


def get_code3d_and_label(df, to_lowercase = False, weighted_graph = False):

    code3d = []
    all_file_label = []
    word_edge = []
    line_edge = []

    for filename, group_df in df.groupby('filename'):

        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        line_lenth = len(group_df)

        code2d, word_adj, line_adj = prepare_code2d(code, line_lenth, to_lowercase, weighted_graph)

        code3d.append(code2d)
        word_edge.append(word_adj)
        line_edge.append(line_adj)
        all_file_label.append(file_label)

    return code3d, all_file_label, word_edge, line_edge


def get_w2v_path():

    return word2vec_dir


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0)
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1, embed_dim)))
    return word2vec_weights


def pad_code(code_list_3d, max_sent_len, limit_sent_len=True, mode='train'):
    paded = []
    
    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)

        if mode == 'train':
            if max_sent_len-len(file) > 0:
                for i in range(0,max_sent_len-len(file)):
                    sent_list.append([0]*max_seq_len)

        if limit_sent_len:    
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)
        
    return paded

def get_x_vec(code_3d, word2vec):
    x_vec = [[[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text]
         for text in texts] for texts in code_3d]
    
    return x_vec