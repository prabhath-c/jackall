def run_slurm(job, queue_name='cmti', num_cores=20, run_time=None, para_mode=None, mpi_proc=None, run=False):
    job.server.queue = queue_name
    job.server.cores = num_cores
    if run_time!=None: job.server.run_time = run_time

    if para_mode!=None:
        job.server.para_mode = para_mode
        if mpi_proc!=None:
            job.server.mpi_proc = mpi_proc

    if run==True: 
        job.run()

    return job