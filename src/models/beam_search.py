import torch
import torch.nn.functional as F


def beam_search(model, x, beam_width=6, device='cuda'):
    batch_size = x.shape[0]

    # each sample in the batch gets its own beam
    beams = [
        Beam(model.vocab, beam_width=beam_width, device=device)
        for _ in range(batch_size)
    ]

    # loop through samples in the batch
    for b in range(batch_size):
        sample = x[b].unsqueeze(0)  # add dummy batch dim
        beam = beams[b]

        src = sample.transpose(0, 1)  # max_seq_len, 1, d_model
        src_key_padding_mask = sample.sum(
            dim=-1) == 0  # pad input if all d_model dims are 0

        while True:  # keep predicting until the beam has finished
            y = beam.decoded  # (active_beams, seq_len)
            tgt = model.embedding(y).transpose(
                0, 1)  # (seq_len, active_beams, d_model)

            # all beams are ran in parallel so expand src batch dim to the
            # number of active beams (this has to be target shape rather than
            # beam.active_beams because on the first iteration beam.decoded has
            # batch size of 1 even though active_beams == beam_width)
            src_expanded = src.expand(-1, tgt.shape[1], -1)
            src_key_padding_mask_expanded = src_key_padding_mask.expand(
                tgt.shape[1], -1)

            out = model.transformer(
                src_expanded,
                tgt,
                src_key_padding_mask=src_key_padding_mask_expanded
            )  # (seq_len, 1, d_model)

            # squeeze the dummy batch dimension
            out = out.squeeze(1)

            next_token_scores = F.log_softmax(model.linear(out[-1]),
                                              dim=-1)  # (1, n_vocab)

            done = beam.advance(next_token_scores)

            if done:
                break

    return [max(beam.finished, key=lambda t: t[1])[0] for beam in beams]


class Beam:
    def __init__(self, vocab, beam_width, device='cuda'):

        self.decoded = torch.tensor([vocab.token2idx('<sos>')
                                     ]).view(1, 1).to(device)
        self.scores = torch.zeros((1, 1)).to(device)
        self.vocab = vocab
        self.active_beams = beam_width
        self.max_seq_len = 200

        # fall back to this default string if eos is not encountered until
        # max_seq_len is reached
        self.finished = [('#FAILED TO DECODE#', -float('inf'))]

    def advance(self, next_token_scores):
        '''next_token_scores: (1, n_vocab)'''

        # self.scores is (active_beams, n_vocab) and next_token_scores is (1,
        # n_vocab)
        beam_scores = self.scores + next_token_scores

        def lp(length, beta=0.6):
            '''
            computes length normalization factor where length is the sequence
            length and beta is a hyperparam
            '''
            return ((5 + length) / 6)**beta

        decoded_length = self.decoded.shape[1]
        length_normalized_scores = beam_scores / lp(
            decoded_length)  # TODO: add shallow fusion here

        topv, topi = length_normalized_scores.view(-1).topk(
            k=self.active_beams)

        beam_idx = topi // self.vocab.n_output
        token_idx = topi % self.vocab.n_output

        is_finished = token_idx == self.vocab.token2idx('<eos>')

        if is_finished.any():
            self.active_beams -= is_finished.sum()

            # if sequence is finished then add it to the list of
            # finished candidates
            finished_sequences = self.decoded.index_select(
                dim=0, index=beam_idx[is_finished])  # TODO: maybe add <eos>?
            finished_scores = length_normalized_scores.index_select(
                dim=0, index=beam_idx[is_finished]).gather(
                    dim=1, index=token_idx[is_finished].reshape(-1, 1))

            for seq, score in zip(finished_sequences, finished_scores):
                # ignore <sos> and map indices to characters
                decoded_sequence = ''.join(
                    self.vocab.idx2token(idx) for idx in seq[1:])

                self.finished.append((decoded_sequence, score.item()))

        if not is_finished.all():
            # if sequence is not finished then add it to decoded and
            # continue decoding
            self.decoded = torch.cat([
                self.decoded.index_select(dim=0, index=beam_idx[~is_finished]),
                token_idx[~is_finished].reshape(-1, 1)
            ],
                                     axis=1)

            # update scores with unnormalized scores that correspond
            # to the top-k beams
            self.scores = beam_scores.index_select(
                dim=0, index=beam_idx[~is_finished]).gather(
                    dim=1, index=token_idx[~is_finished].reshape(-1, 1))

        # stop when all beams are finished or seq len is > 200
        done = self.active_beams < 1 or decoded_length > self.max_seq_len

        return done
