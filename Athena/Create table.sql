create table order_products_prior with (external_location='s3://imba-andy/features/order_products_prior/', format = 'parquet')

as(
select a.*, 
       b.product_id, 
       b.add_to_cart_order,
       b.reordered
from "prd"."orders" a 
     join
     "prd"."orders_products" b
     on a.order_id = b.order_id
where a.eval_set = 'prior');