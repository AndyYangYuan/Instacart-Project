######### create project ###############################################

# create project
library('ProjectTemplate')
create.project('imba')

# load the project
library(ProjectTemplate)
load.project()

# library import
library(reshape2)
library(tidyverse)
library(stringr)
library(lubridate)
library(dplyr)
library(pROC)
library(xgboost)
library(precrec)


######### feature engineering ##########################################

# change names of dataframe
order.products.train <- order_products__train
order.products.prior <- order_products__prior

rm(order_products__train)
rm(order_products__prior)

# append user_id to order.products.train
order.products.train <- order.products.train %>% 
  inner_join(orders[,c("order_id", "user_id")])

# add more features to data
data$prod_reorder_probability <- data$prod_second_orders / data$prod_first_orders
data$prod_reorder_times <- 1 + data$prod_reorders / data$prod_first_orders
data$prod_reorder_ratio <- data$prod_reorders / data$prod_orders
data <- data %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)
data$user_average_basket <- data$user_total_products / data$user_orders
data$up_order_rate <- data$up_order / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_order / (data$user_orders - data$up_first_order + 1)


# select only train and test orders (filter out the prior data of orders)
us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set)

data <- data %>% inner_join(us)

rm(us)
gc()

# creating final data set: adding target variables
data <- data %>% 
  left_join(order.products.train %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()


######### creating training and testing set #########################

train <- data[data$eval_set == 'train',]
# testing data here is actually the forecasting data which will be submitted to kaggle
test <- data[data$eval_set == "test",]


# Delete columns for training data set
train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
# training target variable creation
train$reordered[is.na(train$reordered)] <- 0

# Delete columns for testing data set
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

# split training data into training and validation set
set.seed(1)
# the whole data set of training data
train_bak <- train

# sampling: only sample 10 percent of data for performance reasons
train <- train[sample(nrow(train)),] %>% 
  sample_frac(0.1)


######### xgboost modeling #############################################

# Set the hypter-parameter tunning set
cat("Hypter-parameter tuning\n")
hyper_grid <-expand.grid(
  nrounds =               c(3000),
  max_depth =             c(6),
  eta =                   c(0.1),
  gamma =                 c(0.7),
  colsample_bytree =      c(0.95),
  subsample =             c(0.75),
  min_child_weight =      c(10),
  alpha =                 c(2e-05),
  lambda =                c(10),
  scale_pos_weight =      c(1)
)

nthread <- 4

# transforming the data set for xgboost mdoel
train_data_x <- data.matrix(select(train, -reordered))
train_data_y <- data.matrix(select(train, reordered))
train_data <- xgb.DMatrix(data = train_data_x, label = train_data_y)
test_data <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))

# Create watch list is only for observing output (doesn't change model early stopping)
watchlist <- list(train = train_data)
model_pos <- 1

set.seed(1)

# running the xgboost mdoel
model <- xgb.train(
  data =                      train_data,
  
  nrounds =                   300,
  max_depth =                 hyper_grid[1, "max_depth"],
  eta =                       hyper_grid[1, "eta"],
  gamma =                     hyper_grid[1, "gamma"],
  colsample_bytree =          hyper_grid[1, "colsample_bytree"],
  subsample =                 hyper_grid[1, "subsample"],
  min_child_weight =          hyper_grid[1, "min_child_weight"],
  alpha =                     hyper_grid[1, "alpha"],
  lambda =                    hyper_grid[1, "lambda"],
  scale_pos_weight =          hyper_grid[1, "scale_pos_weight"],
  
  booster =                   "gbtree",
  objective =                 "binary:logistic",
  eval_metric =               "auc",
  prediction =                TRUE,
  verbose =                   TRUE,
  watchlist =                 watchlist,
  early_stopping_rounds =     50,
  print_every_n =             10,
  nthread =                   nthread
)


######### Model validation #############################################

# make a prediction on the training dataset (this is for validation)
pred_train_data <- predict(model, newdata = train_data_x)

# check the performance of the model on training dataset
train_performance <- roc(as.vector(train_data_y), pred_train_data)
train_performance

# plot ROC curve and precision-recall curve
precrec_obj <- evalmod(scores = pred_train_data, labels = as.vector(train_data_y))
autoplot(precrec_obj)

# plot the probability distribution
df <- data.frame(scores = pred_train_data, labels = as.vector(train_data_y))
ggplot(df, aes(x=scores, fill=as.factor(labels))) + geom_density(alpha = 0.5)


######### Forecasting ##################################################

test$reordered <- predict(model, newdata = test_data)
test$reordered <- (test$reordered > 0.21) * 1

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)


#### modeling training validation split, use cv ############

# split data into train and validation
train_index <- sample(1:nrow(train), 0.8 * nrow(train))
# valid_index <- setdiff(1:nrow(data), train_index)

# Create Hypter-parameter tuning set
cat("Hypter-parameter tuning\n")
hyper_grid <-expand.grid(
  nrounds =               c(3000),
  objective =             c("binary:logistic"),
  eval_metric =           c("auc"),
  max_depth =             c(6),
  eta =                   c(0.1,0.05),
  gamma =                 c(0.7),
  colsample_bytree =      c(0.95),
  subsample =             c(0.75),
  min_child_weight =      c(10),
  alpha =                 c(2e-05),
  lambda =                c(10),
  scale_pos_weight =      c(1)
)

nthread <- 4

gc()


cat("Running model tuning\n")

# recording the performance of each parameter set
final_valid_metrics <- data.frame()

# running the xgboost model 
for(i in 1:nrow(hyper_grid)){
  cat(paste0("\nModel ", i, " of ", nrow(hyper_grid), "\n"))
  cat("Hyper-parameters:\n")
  print(hyper_grid[i,])
  
  metricsValidComb <- data.frame()
  
  
  cv.nround = 100
  cv.nfold = 3
  # cross validation
  mdcv <- xgb.cv(data=train_data, params = list(
                                    objective =                 hyper_grid[1, "objective"],
                                    eval_metric =               hyper_grid[1, "eval_metric"],
                                    max_depth =                 hyper_grid[1, "max_depth"],
                                    eta =                       hyper_grid[1, "eta"],
                                    gamma =                     hyper_grid[1, "gamma"],
                                    colsample_bytree =          hyper_grid[1, "colsample_bytree"],
                                    subsample =                 hyper_grid[1, "subsample"],
                                    min_child_weight =          hyper_grid[1, "min_child_weight"],
                                    alpha =                     hyper_grid[1, "alpha"],
                                    lambda =                    hyper_grid[1, "lambda"],
                                    scale_pos_weight =          hyper_grid[1, "scale_pos_weight"]
                          ), nthread=4, nfold=cv.nfold, nrounds=cv.nround, verbose = T)
  
  model_auc = min(mdcv$evaluation_log$test_auc_mean)

  # put together AUC and the best iteration value
  metrics_frame <- data.frame(AUC = model_auc)
  # combine the result for each fold
  #metricsValidComb <- rbind(metricsValidComb, metrics_frame)
  final_valid_metrics <- rbind(final_valid_metrics, metrics_frame)
  cat(paste0("AUC: ", round(model_auc, 3), "\n"))
  
}


results_valid <- cbind(hyper_grid, final_valid_metrics)

# descending on AVG_AUC and get the best parameter
results_valid <- results_valid %>% 
  arrange(desc(AUC))


############## FINAL MODEL #############################################

# use the best model we find to validate the training data

#train_data <- train[-(valid_fold_start_index:valid_fold_end_index),]
train_data_x <- data.matrix(select(train, -reordered))
train_data_y <- data.matrix(select(train, reordered))
train_data <- xgb.DMatrix(data = train_data_x, label = train_data_y)
test_data <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))

# Create watch list
# watchlist <- list(train = train_data)
model_pos <- 1

# run xgboost model
model <- xgb.train(
  data =                      train_data,
  
  nrounds =                   cv.nround,
  max_depth =                 results_valid[model_pos, "max_depth"],
  eta =                       results_valid[model_pos, "eta"],
  gamma =                     results_valid[model_pos, "gamma"],
  colsample_bytree =          results_valid[model_pos, "colsample_bytree"],
  subsample =                 results_valid[model_pos, "subsample"],
  min_child_weight =          results_valid[model_pos, "min_child_weight"],
  alpha =                     results_valid[model_pos, "alpha"],
  lambda =                    results_valid[model_pos, "lambda"],
  scale_pos_weight =          results_valid[model_pos, "scale_pos_weight"],
  
  booster =                   "gbtree",
  objective =                 "binary:logistic",
  eval_metric =               "auc",
  prediction =                TRUE,
  verbose =                   TRUE,
  watchlist =                 watchlist,
  print_every_n =             10,
  nthread =                   nthread
)


# make a prediction on the training dataset
pred_train_data <- predict(model, newdata = train_data_x)

# check the performance of the model on training dataset
train_performance <- roc(as.vector(train_data_y), pred_train_data)

# plot ROC curve and precision-recall curve
precrec_obj <- evalmod(scores = pred_train_data, labels = as.vector(train_data_y))
autoplot(precrec_obj)

# Forecasting using the best model 
test$reordered <- predict(model, newdata = test_data)
test$reordered <- (test$reordered > 0.21) * 1

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)

