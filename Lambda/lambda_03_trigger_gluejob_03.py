import boto3
import csv

print('glue_job_03')


def lambda_handler(_event, _context):

# set up the auto trigger in S3 folder "features/log/",if the "succeeded.csv" is existed.
# the glue_job_03 will be triggered automatically

    gluejob = "glue_job_03"
    glue = boto3.client('glue')
    gluejobname = gluejob
    sns = boto3.client('sns')
    phonenumber = 'input your phone number'
    # s3 = boto3.client('s3')
    # BUCKET_NAME ='imba'
    # prefix ='/features/log/'
    # file_name = 'succeeded.csv'
    # Key = prefix + file_name

    try:
        runId = glue.start_job_run(JobName=gluejobname)
        status = glue.get_job_run(JobName=gluejobname, RunId=runId['JobRunId'])
        print("Job Status : ", status['JobRun']['JobRunState'])
        job_status = status['JobRun']['JobRunState']
    except Exception as e:
        print(e)
        raise
# could add a sns notification in here, when jobrunstate is ok , or not ok , send an email to the designated email box
    finally:
        if job_status == "SUCCEEDED":
            
            # temp_csv_file = csv.writer(open("/tmp/succeeded.csv", "w+"))
            # s3.upload_file('/tmp/csv_file.csv', BUCKET_NAME,Key)
            sns.publish(
                        PhoneNumber = phonenumber,
                        Message = 'Hello, the job status is {}, the ETL_2 will be triggered automatically'.format(job_status)
                    )
        else:
            sns.publish(
            PhoneNumber = phonenumber,
            Message = 'Hello, the job status is {}, the ETL_2 will be not be triggered .pls invenstigate'.format(job_status)
        )
