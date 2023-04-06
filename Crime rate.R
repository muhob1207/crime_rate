library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue)
library(h2o) 
library(purrr)

df <- fread('crimes.csv')
df %>% view()

#All of our variables are numerical so we don't need to get dummies
df %>% skim()

#Inspecting NA. We don't have any NA values
df %>% inspect_na()

#Creating GLM
target <- 'ViolentCrimesPerPop'
features <- df %>% select(-ViolentCrimesPerPop) %>% names()
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

#Removing aliases
coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

#Q1 Find multicollinearity by applying VIF;
while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 1.5){
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df)
}
glm %>% faraway::vif() %>% sort(decreasing = T) %>% names() -> features 
df <- df %>% select(ViolentCrimesPerPop,features)

#Q2 Standardize features;
df <- as.data.frame(df)
df[,-1] <- df[,-1] %>% scale() %>% as.data.frame() 

#Q3 Split data into train and test sets using seed=123;
h2o.init()
h2o_data <- df %>% as.h2o()
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]
target <- 'ViolentCrimesPerPop'
features <- df %>% select(-ViolentCrimesPerPop) %>% names()

#Q4 Build linear regression model. p value of variables should be max 0.05;
model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

#Doing the stepwise backward elimination and retraining the model
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      dplyr::select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] > 0.05) {
  model@model$coefficients_table %>%
    as.data.frame() %>%
    dplyr::select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  test_h2o <- test %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target,
    training_frame = train,
    validation_frame = test,
    nfolds = 10, seed = 123,
    lambda = 0, compute_p_values = T)
}

#We have now removed all the columns for which the p-value is higher than 0.05
#Our model now has 6 features left.
model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) 

#Making predictions
y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred_train <- model %>% h2o.predict(newdata = train) %>% as.data.frame()
train_set <- train %>% as.data.frame()
test_set <- test %>% as.data.frame()

#Q5 Calculate RMSE and Adjusted R-squared;
residuals = test_set$ViolentCrimesPerPop - y_pred$predict

RMSE = sqrt(mean(residuals^2))

y_test_mean = mean(test_set$ViolentCrimesPerPop)

tss = sum((test_set$ViolentCrimesPerPop - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

# The model has performed very bad
tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)

#Q6 Check overfitting.
library(patchwork)

#Making a plot for test data
my_data <- cbind(predicted = y_pred$predict,
                 observed = test_set$ViolentCrimesPerPop) %>% 
  as.data.frame()

p1 <- my_data %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Output", 
       y="Observed Output",
       title=glue('Adjusted R2 = {round(enexpr(Adjusted_R2),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))


#Calculating residuals for train data
residuals = train_set$ViolentCrimesPerPop - y_pred_train$predict

RMSE_train = sqrt(mean(residuals^2))
y_train_mean = mean(train_set$ViolentCrimesPerPop)

tss = sum((train_set$ViolentCrimesPerPop - y_train_mean)^2)
rss = sum(residuals^2)

R2_train = 1 - (rss/tss); R2_train

n <- train_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2_train = 1-(1-R2_train)*((n-1)/(n-k-1))

#Plotting the results for train data
my_data_train <- cbind(predicted = y_pred_train$predict,
                       observed = train_set$ViolentCrimesPerPop) %>% 
  as.data.frame()

g_train <- my_data_train %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Output", 
       y="Observed Output",
       title=glue('Adjusted R2 = {round(enexpr(Adjusted_R2_train),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

#Comparing the plots for training and testing data
g_train + p1


#The model performed better on test data so there is no overfitting
tibble(RMSE_train = round(RMSE_train,1),
       RMSE_test = round(RMSE,1),
       
       Adjusted_R2_train,
       Adjusted_R2_test = Adjusted_R2)
