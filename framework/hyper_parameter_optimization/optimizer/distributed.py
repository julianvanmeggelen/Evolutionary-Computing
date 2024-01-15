from typing import Any
from hyper_parameter_optimization.optimizer.optimizer import OptunaHyperOptimizer
from hyper_parameter_optimization.result.optimization_result import OptimizationResult

from optuna.integration.dask import DaskStorage
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import optuna



def optunaSlurmWorkerMethod(study: optuna.Study, checkpoint_dir: str, optimizer_args: dict[str, Any], run_args: dict[str, Any]):
    tuner = OptunaHyperOptimizer(
        checkpoint_dir = checkpoint_dir,
        **optimizer_args
    )
    result = tuner.run_parallel(study=study,  **run_args)
    return result

class DistributedOptunaSlurmHyperOptimizer():
    def __init__(self, n_workers: int, checkpoint_dir: str,  cores = 16, memory = '64GB', db:str = None, **kwargs) -> None:
        self.n_workers = n_workers
        self.optimizer_args = kwargs
        self.result = None
        self.cores  = cores
        self.db = db
        self.checkpoint_dir = checkpoint_dir
        self.memory = memory

    def run(self, **kwargs):
        cluster = SLURMCluster(
            cores=self.cores,  # Adjust as needed
            memory =self.memory,
            processes = 1,
            nanny=False,
            walltime='105:00:00',
            #job_extra_directives = ['--time=0-105:00:00']
        )

        # Scale the cluster to the desired number of workers
        cluster.scale(self.n_workers)  # Adjust as needed

        # Connect a Dask client to the cluster
        client = Client(cluster)

        dask_storage = DaskStorage(self.db)
        study = optuna.create_study(storage=dask_storage)

        #merge result objects
        jobs = []
        for i in range(self.n_workers): 
            checkpoint_dir = self.checkpoint_dir + '_' + str(i)
            res = client.submit(optunaSlurmWorkerMethod, study, checkpoint_dir, self.optimizer_args, kwargs, key = str(i)) 
            jobs.append(res)
        final_result = client.gather(jobs)
        final_result = OptimizationResult.merge(list(final_result))

        self.result = final_result
        return final_result
