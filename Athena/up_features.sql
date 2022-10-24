select user_id,
       product_id,
       count(distinct order_id) as up_orders,
       min(order_number) as up_first_order,
       max(order_number) as up_last_order,
       round(avg(add_to_cart_order),0) as up_average_cart_position
from "prd"."order_products_prior"
group by user_id, product_id;