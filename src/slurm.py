
from logging import getLogger
import os
import sys
import torch
import socket
import signal
import subprocess


logger = getLogger()

def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    #logger.warning("Signal handler installed.")


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ 
    has_local_rank = hasattr(params, 'local_rank')

    # SLURM job
    if params.is_slurm_job and has_local_rank:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            #print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        # params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.main_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.main_port <= 20000 or params.world_size == 1
        #print(PREFIX + "Master address: %s" % params.master_addr)
        #print(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.main_addr
        os.environ['MASTER_PORT'] = str(params.main_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)
        params.is_distributed = True


    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif has_local_rank and params.local_rank != -1:

        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.is_distributed = True

    else:
        n_gpu = torch.cuda.device_count()
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = n_gpu
        params.n_gpu_per_node = n_gpu
        params.is_distributed = False

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device


    # initialize multi-GPU
    if params.is_distributed:

        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        #print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )