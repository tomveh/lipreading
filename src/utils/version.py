import os
from datetime import datetime


def version(version_hparams, hparams):
    v = f'version_{os.environ["SLURM_JOB_ID"]}'

    for (display, key) in version_hparams:
        if key is None:
            key = display

        v += f'_{display}={str(vars(hparams)[key]).replace(" ", "-")}'

    v += '_date=' + datetime.now().strftime("%b-%d-%H:%M")

    return v
