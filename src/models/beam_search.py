import torch
import torch.nn.functional as F


def _lp(length, beta=0.6):
    '''
    computes length normalization factor where length is the sequence
    length and beta is a hyperparam
    '''
    return ((5 + length) / 6)**beta


def beam_search(model,
                x,
                beam_width=6,
                batch_size=1000,
                max_seq_len=100,
                device='cuda',
                min_finished=10):
    N, S, E = x.shape

    results = [[] for _ in range(N)]

    # all beams start with <sos> but this will later become
    # (N * beam_width, decoded_sequence_length)
    decoded = torch.tensor([model.vocab.token2idx('<sos>')
                            ]).expand(N, 1).clone().to(device)

    # unnormalized scores for decoded sequences
    # same shape as decoded but without trailing dimension
    scores = torch.zeros(N, 1).to(device)

    while True:
        split_log_probs = []

        # on first iteration all beams are the same so no replication
        beam_dim_size = 1 if decoded.shape[-1] == 1 else beam_width

        # expand data to run beams in parallel
        x_expanded = x.repeat(1, beam_dim_size, 1).reshape(-1, S, E)

        # split data to not run out of memory
        for x_split, decoded_split in zip(x_expanded.split(batch_size),
                                          decoded.split(batch_size)):
            src = x_split.transpose(0, 1).contiguous()  # S, B, E
            tgt = model.embedding(decoded_split).transpose(
                0, 1).contiguous()  # T, B, E

            tgt_mask = model.transformer.generate_square_subsequent_mask(
                len(tgt)).type_as(tgt)

            src_key_padding_mask = (x_split == 0).all(dim=2)
            tgt_key_padding_mask = decoded_split == model.vocab.token2idx(
                '<pad>')

            # return shape is (S, B, E)
            out = model.transformer(src,
                                    tgt,
                                    tgt_mask=tgt_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    src_key_padding_mask=src_key_padding_mask)

            split_lprobs = F.log_softmax(model.linear(out[-1]), dim=1)
            split_log_probs.append(split_lprobs.detach())

        # concatenate all log probs
        log_probs = torch.cat(split_log_probs).reshape(N, beam_dim_size, -1)

        # scores for every beam and all possible next predictions
        # shape: batch x beam_dim_size x n_vocab
        beam_scores = scores.unsqueeze(-1) + log_probs

        # length normalization
        normalized_scores = beam_scores / _lp(decoded.shape[-1] - 1)

        # normalized scores are used to pick the next token
        _, topi = normalized_scores.reshape(N, -1).topk(k=beam_width, dim=1)

        beam_idx = topi // model.vocab.n_output
        token_idx = topi % model.vocab.n_output

        decoded_beginning = torch.cat([
            decoded.reshape(N, beam_dim_size,
                            -1)[i].index_select(dim=0, index=idx).unsqueeze(0)
            for i, idx in zip(range(N), beam_idx)
        ])

        # update decoded
        decoded = torch.cat(
            [decoded_beginning, token_idx.unsqueeze(-1)],
            dim=2).reshape(N * beam_width, -1)

        # update scores
        scores = normalized_scores.reshape(
            N, beam_dim_size * model.vocab.n_output).gather(
                dim=1, index=topi).reshape(N, beam_width)

        # beam is finished if we predicted <eos>
        finished = token_idx == model.vocab.token2idx('<eos>')

        # if beam finished then append the result to results

        for batch_i, beam_i in finished.nonzero():
            indices = decoded.reshape(N, beam_width, -1)[batch_i, beam_i][1:-1]

            vocab_cls_name = model.vocab.__class__.__name__
            assert vocab_cls_name in ['SubwordVocab', 'CharVocab']

            if vocab_cls_name == 'SubwordVocab':
                seq = model.vocab.decode(list(indices))
            else:
                seq = ''.join([
                    model.vocab.idx2token(idx)
                    for idx in decoded.reshape(N, beam_width, -1)[batch_i,
                                                                  beam_i][1:-1]
                ])

            score = scores[batch_i, beam_i].item()
            results[batch_i].append({'seq': seq, 'score': score})

            # set beam score to -inf so we won't pick this sequence again
            scores[batch_i, beam_i] = float('-inf')

        # stop if max length reached or every beam has at least
        # min_finished results
        if min([len(r) for r in results
                ]) > min_finished or decoded.shape[-1] > max_seq_len:
            break

    return results
