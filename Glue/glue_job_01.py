import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job



def main():

    # create sparkcontext, gluecontext and glue-spark_session first
    sc =SparkContext.getOrCreate()
    glueContext = GlueContext(sc)
    spark =glueContxt.spark_session

    #parameters
    glue_database = 'database'
    first_ETL = 'imba/features/order_products_prior/'
    
    
    # option 1 creating dataframes directly from the S3 bucket
    # orders = glueContext.create_dynamic_frame_from_options(connection_type = "csv", connection_options = {"paths": ["s3://--bucket_name--/features/order/"]})
    
        
    #  option 2 creating dataframes from existing glue catelog

    orders = glueContext.create_dynamic_frame.from_catalog(database = {}, table_name = 'orders').format(glue_database)
    order_products= glueContext.create_dynamic_frame.from_catalog(database = {}, table_name = 'order_products').format(glue_database)
    
       
    # convert glue dynamic dataframe to spark dataframe first , then to the Tempview and table 
     orders.toDF().createOrReplaceTempView('orders')
     order_products.toDF().createOrReplaceTempView('order_products')

    # run your spark.sql() on the TempView
    # the returned value is a spark dataframe, if need to be further reused in another spark sql, you need to covert it into TempView again

    order_products_prior = spark.sql('''
                                    SELECT a.*, 
                                    b.product_id, 
                                    b.add_to_cart_order, 
                                    b.reordered 
                                    FROM orders a 
                                    JOIN order_products b 
                                    ON a.order_id = b.order_id 
                                    WHERE a.eval_set = 'prior'
                                    ''')

    # write your newly generated dynamic frame  back to S3 bucket 

    order_products_prior.repartition(1).write.mode('overwrite').format('parquet').save("s3://{}", header = 'true').format(first_ETL)
    
if __name__ == '__main__':
    main()