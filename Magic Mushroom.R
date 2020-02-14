
# MAGIC MUSHROOM ----------------------------------------------------------
# GOAL : Train model to detect poisonous mushrooms 
#
# DESCRIPTION : In this script, I develop a Model to predict whether mushrooms are poisonous.
# The Model should be based only on characteristics that can be observed by a camera 
# (It is for an app). Further, the main difficulty of this model is an imbalanced data set.
# That is to say, there are far more eatable than poisonous mushrooms. As a consequence, a 
# classification model would be biased towards 'eatable'. I solve this issue by oversampling
# observations from the poisonous class, and balance out the data set accordingly. 
# More specifically, I resampled false negatives until everybody survived.
# Thus, also note, the main performance metric here is not accuracy, but rather recall.
# That is, the model should not classify any poisonous mushrooms as eatable, 
# as that would result in painful death.
# 
# DEVELOPER : Berend 
# Thu Jan 16 11:37:33 2020 ------------------------------


# Pacman -------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load("tidyverse", "e1071", "dbplyr", "modelr",
                "caret", "data.table", "pls", "magrittr",
               "MLmetrics")

# Load Data ---------------------------------------------------------------
mush <- readRDS("Data/train.rds")


# Make factors and df -----------------------------------------------------
mush <- mush %>% apply(2, as.factor)
mush <- data.frame(mush)

# Feature selection and data cleaning -------------------------------------
# list of the relevant features (do not ask me why)
relevant_feat <- c("cap.shape","cap.color", "stalk.color.above.ring",
                  "stalk.color.below.ring", "class","bruises","population")

# keep only them 
mush <- mush[, relevant_feat]

# Keep only unique observations
# from 6093 to 398
mush %<>% unique()


# Check distribution of e/p -----------------------------------------------

# classes:                 e    p 
# whole set :           3156 2937
# unique observations :  244  154

# code for numeric representation of distribution:
#       table(mush$class)
# and for visualisation:
#       histogram(mush$class, type = "count")


# Create train & test set -------------------------------------------------
set.seed(123)

# Data Partition 
inTrain <- createDataPartition(y = mush$class, p = .75, list = FALSE)
# create training/test set
mush_train <- mush[inTrain, ]
mush_test <- mush[-inTrain, ]
 
# Train Model -------------------------------------------------------------------------
# modify resampling method
ctrl <- trainControl(method = "repeatedcv", 
                     verboseIter =  FALSE,
                     repeats = 1,
                     number = 4, 
                     classProbs = TRUE)

# train model : knn for speed
# Note : rf is probably a better algo for this problem
my_model <- train(class ~ .,
                  mush_train,
                  method = "knn",
                  tuneGrid = expand.grid(k = c(2, 3, 4)),
                  trControl = ctrl
                  )

# run predictions
predictions <-predict(my_model, mush_test)
# Create confusion matrix
cm <- confusionMatrix(predictions,
                mush_test$class,
                mode = "prec_recall"
                )

# print(cm$table)
##### prSummary(test_set, lev = levels(test_set$obs))
print(cm$table)
print(cm$byClass["Recall"])

# Identify errors ---------------------------------------------------------
errors <- 
  mush %>% 
  add_predictions(model = my_model, 
                  var = "pred") %>% 
  filter(class != pred & class == "p")

# adding the errors in our data to create better model
mush_resample <- 
  mush %>% 
  bind_rows(errors) %>%
  select(-pred)


# Feed the errors back into the data until there are no more deaths -------

# Set value for the start of a while loop
# deaths to a positive number, we want the loop to run until deaths = 0
dead <- 1
# index counts the number of loops, start at 1
index <- 1
# create empty df to store errors for later analysis
errors_df <- tibble()

# while loop : as long as people die
# create new model models, based on the resampled data sets
while (dead != 0) {
  
  # create data partition
  inTrain <- createDataPartition(y = mush_resample$class,
                                 p = 0.75,
                                 list = F
                                 )
  
  # create train/test sets
  re_train <- mush_resample[inTrain, ]
  re_test  <- mush_resample[-inTrain, ]
  
  # train model on renewed data 
  re_model <- train(class ~ .,
                 data = re_train,
                 method = "knn",
                 tuneGrid = expand.grid(k = c(2, 3, 4)),
                 trControl = ctrl
                 )
  
  # run predictions
  re_predictions <- predict(re_model, re_test)
  
  # Create confusion matrix
  re_cm <- confusionMatrix(data = re_predictions,
                           reference = re_test$class,
                           mode = "prec_recall"
                           )
  
  # print results
  print(paste("Try #", index, "... ---------------------------------------------------------------",
              sep = ""),
        quote = F)
  print(re_cm$table)
  print(re_cm$byClass["Recall"])


  # identify errors
  mush_errors_loop <- mush_resample %>% 
    add_predictions(model = re_model, var = "pred") %>% 
    filter(class != pred & class == "p")
  
  # count them
  dead <- nrow(mush_errors_loop)
  # print results
  if (dead != 0) {
    print(paste("You killed ", dead, "!", sep = ""), quote = F)
  } else { 
    print("Hurray, everybody lives!", quote = F)
  }
    
  # add the errors to main data
  mush_resample <- mush_resample %>% 
    bind_rows(mush_errors_loop) %>% 
    select(-pred)
  
  # store errors in errors_df
  errors_df <- errors_df %>% 
    bind_rows(mush_errors_loop %>% 
                mutate(index = index))
  
  # index + 1 and reloop
  index = index + 1
}


# Everybody lives! save that model ----------------------------------------
write_rds(re_model, path = "Data/my_mushroom_model.rds")




