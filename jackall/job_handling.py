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

def run_jobs_from_dataframe(df, job_column_name='job', submit=True, slurm_details={}):
    for _, row in df.iterrows():
        job = row[job_column_name]

        if submit==True:
            try:
                run_slurm(job, **slurm_details)
            except:
                None
        elif submit==False:
            job.run()