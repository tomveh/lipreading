import torch.nn as nn

from .backends import ConvolutionalBackend, TransformerBackend
from .frontend import VisualFrontend

# class LSTMLangModel(nn.Module):
#     def __init__(self, vocab):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings=vocab.n_embed,
#                                       embedding_dim=1024,
#                                       padding_idx=vocab.token2idx('<pad>'))
#         self.lstm = nn.LSTM(input_size=1024,
#                             hidden_size=1024,
#                             num_layers=4,
#                             batch_first=True,
#                             dropout=0,
#                             bidirectional=False)
#         self.linear = nn.Linear(1024, vocab.n_output)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         x, _ = self.lstm(self.embedding(x))
#         return self.softmax(self.linear(x))


class PretrainNet(nn.Module):
    def __init__(self, resnet, nh):
        super().__init__()
        self.frontend = VisualFrontend(out_channels=nh, resnet=resnet)
        self.backend = ConvolutionalBackend(nh, 500)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab, resnet, nh):
        super().__init__()
        self.vocab = vocab
        self.frontend = VisualFrontend(out_channels=nh, resnet=resnet)
        self.backend = TransformerBackend(vocab, nh)

    def forward(self, x, y=None):
        x = self.frontend(x.unsqueeze(1))
        x = self.backend(x, y)
        return x


# class TransformerModel(nn.Module):
#     def __init__(
#             self,
#             vocab,
#             nh,
#     ):
#         super().__init__()
#         self.frontend = VisualFrontend(out_channles=nh)
#         self.backend = TransformerBackend(vocab=vocab, nh=nh)

#     def forward(self, x, y=None):

#         if self.training and y is not None:

#             # visual_features = self.frontend(x)
#             visual_features = x

#             y_embedded = self.embedding(y)

#             # nn.Transformer wants shapes (S, N, E)...
#             src = visual_features.transpose(0, 1).contiguous()
#             # ...and (T, N, E)
#             tgt = y_embedded.transpose(0, 1).contiguous()

#             # not allowed to look ahead
#             tgt_mask = self.transformer.generate_square_subsequent_mask(
#                 len(tgt)).to(self.device)

#             src_key_padding_mask = x.sum(dim=2) == 0
#             tgt_key_padding_mask = y == self.vocab.token2idx('<pad>')

#             out = self.transformer(src,
#                                    tgt,
#                                    tgt_mask=tgt_mask,
#                                    src_key_padding_mask=src_key_padding_mask,
#                                    tgt_key_padding_mask=tgt_key_padding_mask)

#             pred = self.linear(out)

#             # seq_len x batch_size x n_vocab -> batch_size x n_vocab x
#             # seq_len_n_vocab
#             pred = pred.permute(1, 2, 0).contiguous()

#             return pred

#         elif not self.training and y is None:
#             return beam_search(self, self.vocab, x, beam_width=10)

#         else:
#             raise RuntimeError('not sure if train or eval')

# def inference(self, x):
#     # visual_features = self.frontend(x)
#     visual_features = x

#     sos = self.vocab.token2idx('<sos>')
#     pad = self.vocab.token2idx('<pad>')
#     eos = self.vocab.token2idx('<eos>')

#     batch_size = len(visual_features)

#     # every prediction starts with <sos>
#     y = torch.tensor([sos] * batch_size).view(batch_size,
#                                               1).to(self.device)

#     src = visual_features.transpose(0, 1).contiguous()
#     # src_key_padding_mask = x.sum(dim=(-1, -2)).squeeze() == 0
#     src_key_padding_mask = x.sum(
#         dim=-1) == 0  # TODO this is only visual features, no raw input

#     # for each sequence in batch keep track if <eos> (indicates end of
#     # sequence) has been seen
#     done = torch.tensor([False] * batch_size).to(self.device)

#     seq_len = 0
#     max_seq_len = 100

#     while not done.all() and seq_len <= max_seq_len:

#         y_embedded = self.embedding(y)
#         tgt = y_embedded.transpose(0, 1).contiguous()

#         out = self.transformer(src,
#                                tgt,
#                                src_key_padding_mask=src_key_padding_mask)

#         # transformer outputs (T, N, E)
#         # where T = target seq len, N = batch size, E = embedding dim size

#         # let's take the last char of the predicted sequence for each
#         # batch: out[-1, :, :]
#         pred = F.log_softmax(self.linear(out[-1]), dim=-1)
#         # pred is of shape (N, n_vocab)

#         values, indices = pred.max(dim=-1, keepdim=True)

#         # indices is of shape (batch_size, 1) and contains
#         # the index for the best token

#         # due to batch processing we get prediction even for those samples
#         # where we have already encountered <eos>. Let's replace the
#         # prediction with <pad> if <eos> has been predicted earlier
#         indices[done] = pad

#         y = torch.cat([y, indices], dim=1).to(self.device)

#         seq_len += 1

#         done = done | (indices.squeeze() == eos)

#     return y

# def beam_search(self, x, beam_width):
#     sos = self.vocab.token2idx('<sos>')
#     pad = self.vocab.token2idx('<pad>')
#     eos = self.vocab.token2idx('<eos>')

#     if len(x.shape) == 5:
#         # raw data - batch x channel x seq x width x height
#         batch_size, __, max_seq_len, __, __ = x.shape
#         visual_features = self.frontend(x)
#     elif len(x.shape) == 3:
#         # preprocessed visual features - batch x seq x d_model
#         batch_size, max_seq_len, __ = x.shape
#         visual_features = x

#     # beam_width hypotheses for each sample.
#     # every hypotheses starts with a <sos> token
#     candidates = torch.tensor([sos]).expand(batch_size, beam_width,
#                                             1).to(self.device)

#     # log prob for each hypothesis is initially 0
#     candidate_scores = torch.FloatTensor([0]).expand(
#         batch_size, beam_width).to(self.device)

#     done = torch.tensor([False]).expand(batch_size,
#                                         beam_width).to(self.device)

#     if len(x.shape) == 5:
#         # sum width and height dims. if sum is 0 then this frame is padding
#         src_mask = x.sum(dim=(-2, -1)) == 0
#     elif len(x.shape) == 3:
#         # add dummy dimension corresponding to color channel and mask if
#         # the features are all 0s
#         src_mask = x.unsqueeze(1).sum(dim=-1) == 0

#     # reuse the color-channel dim and expand it to beam_width. Now we can
#     # feed this to the dataloader and get the same padding mask for each
#     # beam
#     src_mask = src_mask.expand(-1, beam_width,
#                                -1).reshape(batch_size * beam_width, -1)

#     # add dim after batch and extend it to beam width and then collapse
#     # batch and beam to run them in parallel
#     x = visual_features.unsqueeze(1).expand(-1, beam_width, -1,
#                                             -1).reshape(
#                                                 batch_size * beam_width,
#                                                 max_seq_len, -1)

#     seq_len = 0
#     max_seq_len = 100

#     # in this loop keep predicting until we hit <eos> in all batches and
#     # beams or until max_seq_len is reached
#     while not done.all() and seq_len <= max_seq_len:
#         y = candidates.reshape(batch_size * beam_width, -1)
#         y_embedded = self.embedding(y)

#         ds = TensorDataset(x, y_embedded, src_mask)
#         dl = DataLoader(ds, batch_size=8)

#         log_probs = None

#         # the original data has been expanded so that each beam gets it own
#         # sequence. We might not be able to fit batch*beam samples in
#         # memory so let's loop over a dl that handles reasonable number of
#         # batches at time
#         for x_, y_, src_mask_ in dl:

#             src = x_.transpose(0, 1).contiguous()
#             tgt = y_.transpose(0, 1).contiguous()

#             tgt_mask = self.transformer.generate_square_subsequent_mask(
#                 len(tgt)).to(self.device)

#             out = self.transformer(src,
#                                    tgt,
#                                    tgt_mask=tgt_mask,
#                                    src_key_padding_mask=src_mask_)

#             batch_log_probs = F.log_softmax(self.linear(out[-1]), dim=-1)

#             if log_probs is None:
#                 log_probs = batch_log_probs
#             else:
#                 log_probs = torch.cat([log_probs, batch_log_probs], dim=0)

#         # reshape collapsed scores
#         next_token_scores = log_probs.reshape(batch_size, beam_width, -1)
#         n_vocab = next_token_scores.shape[-1]

#         # candidate_probs is implicitly expanded during addition and the
#         # result is (batch, beam, n_vocab)
#         all_candidate_scores = candidate_scores.unsqueeze(
#             -1) + next_token_scores

#         # we do not want to consider scores of sequences where <eos> has
#         # been seen so replace those scores with -inf
#         all_candidate_scores[done] = float('-inf')

#         # it is not enough to find top-1 for each beam in the
#         # vocab dimension because one beam can have multiple good
#         # candidates and some other beam might get completely discarded. We
#         # have to collapse beam and vocab into same dim and search over all
#         # those scores
#         topv, topi = all_candidate_scores.view(batch_size,
#                                                -1).topk(beam_width)
#         # TODO: the problem here is that if multiple beams have the same
#         # beginning they will both pick the continuation that maximizes the
#         # probability i.e. they will always be the same

#         # (batch, n_beam), contains best beam indices for all batches
#         beam_idx = topi // n_vocab

#         # (batch, n_beam), contains topi which tell how to continue
#         # corresponding beam to make it optimal
#         token_idx = topi % n_vocab

#         # if <eos> has been encountered then replace prediction with <pad>
#         # and do not update score
#         token_idx[done] = pad
#         candidate_scores = torch.where(done, candidate_scores, topv)

#         # now we would like to choose the beams corresponding to best
#         # scores. Note that a single beam might be picked multiple times if
#         # it has multiple next tokens that give good score
#         best_beams = candidates.gather(
#             dim=1, index=beam_idx.unsqueeze(-1).expand_as(candidates))

#         candidates = torch.cat(
#             [best_beams, token_idx.unsqueeze(-1)], dim=-1)

#         done = done | (token_idx == eos)

#         # print(seq_len)
#         seq_len += 1

#     k_best_scores, k_best_indices = torch.topk(candidate_scores,
#                                                k=beam_width,
#                                                dim=1)

#     print('best indices', k_best_indices)
#     print('best scores', k_best_scores)

#     # expand last dim to math sequence length to select the whole sequence
#     index = k_best_indices.unsqueeze(-1).expand(-1, -1,
#                                                 candidates.shape[-1])
#     return candidates.gather(dim=1, index=index), k_best_scores

# class Beam:
#     def __init__(self, vocab, beam_width, device='cuda'):
#         self.decoded = torch.tensor([vocab.token2idx('<sos>')
#                                      ]).view(1, 1).to(device)
#         self.scores = torch.zeros((1, 1)).to(device)
#         self.vocab = vocab
#         self.active_beams = beam_width
#         self.done = False
#         self.max_seq_len = 200
#         self.finished = []

#     def advance(self, next_token_scores):
#         '''next_token_scores: (1, n_vocab)'''

#         beam_scores = self.scores + next_token_scores

#         def lp(length, beta=0.6):
#             '''
#             computes length normalization factor where length is the sequence
#             length and beta is a hyperparam
#             '''
#             return ((5 + np.abs(length)) / 6)**beta

#         seq_len = self.decoded.shape[
#             1] + 1  # sequence so far + latest predicted token
#         length_normalized_scores = beam_scores / lp(seq_len)

#         topv, topi = length_normalized_scores.view(-1).topk(
#             k=self.active_beams)

#         beam_idx = topi // self.vocab.n_output
#         token_idx = topi % self.vocab.n_output

#         is_finished = token_idx == self.vocab.token2idx('<eos>')

#         self.active_beams -= is_finished.sum()

#         # if sequence is finished then add it to the list of finished
#         # candidates
#         finished_sequences = self.decoded.index_select(
#             dim=0, index=beam_idx[is_finished])
#         finished_scores = length_normalized_scores.index_select(
#             dim=0, index=beam_idx[is_finished]).gather(
#                 dim=1, index=token_idx[is_finished].reshape(-1, 1))

#         for seq, score in zip(finished_sequences, finished_scores):
#             self.finished.append((seq, score))

#         # if sequence is not finished then add it to decoded and keep decoding
#         self.decoded = torch.cat([
#             self.decoded.index_select(dim=0, index=beam_idx[~is_finished]),
#             token_idx[~is_finished].reshape(-1, 1)
#         ],
#                                  axis=1)

#         # update scores with unnormalized scores that correspond to the top-k
#         # beams
#         self.scores = beam_scores.index_select(
#             dim=0, index=beam_idx[~is_finished]).gather(
#                 dim=1, index=token_idx[~is_finished].reshape(-1, 1))

#         # stop when all beams are finished or seq len is > max_seq_len
#         done = self.active_beams < 1 or self.decoded.shape[1] > self.max_seq_len

#         return done
