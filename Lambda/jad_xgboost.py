import boto3

def lambda_handler(event, context):

    body = event['body']

    # invoke the endpoint
    runtime = boto3.Session().client('sagemaker-runtime')

    # sending the review
    response = runtime.invoke_endpoint(EndpointName = '***ENDPOINT NAME HERE***',
                                       ContentType = 'text/csv',                 
                                       Body = body
                                       )

    # response is an HTTP response
    result = response['Body'].read().decode('utf-8')

    # Round result 
    result = float(result)

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }