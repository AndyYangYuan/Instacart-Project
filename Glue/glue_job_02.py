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
    second_ETL_feature_1 = 'imba/features/user_features_1_db/'
    second_ETL_feature_2 = 'imba/features/user_features_2_db/'
    second_ETL_up_feature = 'imba/features/up_features_db/'
    second_ETL_prd_feature = 'imba/features/prd_features_db/'
    
    #  creating dataframes directly from the S3 bucket
    order_products_prior = glueContext.create_dynamic_frame_from_options(connection_type = "parquet", \
                                                                         connection_options = {"paths": ["s3://{}".format(first_ETL)]}
                                                                         )
    
        
    #  creating dataframes from existing glue catelog

    orders = glueContext.create_dynamic_frame.from_catalog(database = {}, table_name = 'orders').format(glue_database)
    order_products= glueContext.create_dynamic_frame.from_catalog(database = {}, table_name = 'order_products').format(glue_database)
    
       
    # convert glue dynamic dataframe to spark dataframe first , then to the Tempview and table 
     orders.toDF().createOrReplaceTempView('orders')
     order_products_prior.toDF().createOrReplaceTempView('order_products_prior')

    # run your spark.sql() on the TempView
    # the returned value is a spark dataframe, if need to be further reused in another spark sql, you need to covert it into TempView again

    user_features_1 = spark.sql('''
                                    SELECT user_id, 
                                    Max(order_number) AS user_orders, 
                                    Sum(days_since_prior_order) AS user_period, 
                                    Avg(days_since_prior_order) AS user_mean_days_since_prior 
                                    FROM orders
                                    GROUP BY user_id
                                    ''')
    user_features_2 = spark.sql('''
                                    SELECT user_id, 
                                    Count(*) AS user_total_products, 
                                    Count(DISTINCT product_id) AS user_distinct_products ,
                                    Sum(CASE WHEN reordered = 1 THEN 1 ELSE 0 END) / Cast(Sum(CASE WHEN 
                                    order_number > 1 THEN 1 ELSE 0 END) AS DOUBLE) AS user_reorder_ratio
                                    FROM order_products_prior 
                                    GROUP BY user_id
                                    ''')
    up_features= spark.sql('''
                            SELECT user_id, 
                            product_id, 
                            Count(*) AS up_orders, 
                            Min(order_number) AS up_first_order, 
                            Max(order_number) AS up_last_order, 
                            Avg(add_to_cart_order) AS up_average_cart_position 
                            FROM order_products_prior 
                            GROUP BY user_id, 
                            product_id
                              ''')
    prd_features= spark.sql('''
                            SELECT product_id, 
                            Count(*) AS prod_orders, 
                            Sum(reordered) AS prod_reorders, 
                            Sum(CASE WHEN product_seq_time = 1 THEN 1 ELSE 0 END) AS prod_first_orders,
                            Sum(CASE WHEN product_seq_time = 2 THEN 1 ELSE 0 END) AS prod_second_orders
                            FROM (SELECT *, 
                            Rank() 
                            OVER ( 
                            partition BY user_id, product_id 
                            ORDER BY order_number) AS product_seq_time 
                            FROM order_products_prior)
                              ''')
    

    # write your newly generated dynamic frame  back to S3 bucket 

    user_features_1.repartition(1).write.mode('overwrite').format('parquet').save("s3://{}", header = 'true').format(second_ETL_feature_1)
    user_features_2.repartition(1).write.mode('overwrite').format('parquet').save("s3://{}", header = 'true').format(second_ETL_feature_2)
    up_features.repartition(1).write.mode('overwrite').format('parquet').save("s3://{}", header = 'true').format(second_ETL_up_feature)
    prd_features.repartition(1).write.mode('overwrite').format('parquet').save("s3://{}", header = 'true').format(second_ETL_prd_feature)
    
if __name__ == '__main__':
    main()