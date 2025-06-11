def run_slurm(job, queue_name='cmti', num_cores=20, run_time=3600, run=False):
    job.server.queue = queue_name
    job.server.cores = num_cores
    job.server.run_time = run_time

    if run==True: 
        job.run()

    return job