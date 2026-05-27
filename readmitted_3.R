unzip("~/Desktop/R/untitled folder/archive.zip", list = TRUE)

diabetes <- read.csv(unz("~/Desktop/R/untitled folder/archive.zip", "diabetic_data.csv"))

library(naniar)
library(visdat)
library(tidymodels)
library(ggplot2)
library(skimr)

#View(diabetes)

# first clean from the original
diabetes_clean <- diabetes %>% 
  select(-weight, -patient_nbr, -encounter_id, 
         -diag_1, -diag_2, -diag_3, 
         -payer_code, -medical_specialty)

# then replace ? with NA
diabetes_clean[diabetes_clean == "?"] <- NA

# then create binary outcome and drop original
diabetes_clean <- diabetes_clean %>%
  mutate(readmitted = as.factor(ifelse(readmitted == "<30", 1, 0)))

#variables need to be converted to factor from character

diabetes_clean <- diabetes_clean %>%
  mutate(across(where(is.character), as.factor))

levels(diabetes_clean$readmitted)

skim(diabetes_clean)


#non predictive variables were dropped, ? changed to NA


# EDA ---------------------------------------------------------------------
View(diabetes_clean)


ggplot(diabetes_clean, aes(x = num_medications, fill = readmitted)) +
  geom_density(alpha = 0.5) +
  theme_minimal()
#weak

ggplot(diabetes_clean, aes(x = time_in_hospital, fill = readmitted)) +
  geom_density(alpha = 0.5) +
  theme_minimal()


ggplot(diabetes_clean, aes(x = num_lab_procedures, fill = readmitted)) +
  geom_density(alpha = 0.5) +
  theme_minimal()
#weak 


# LASSO setup -------------------------------------------------------------

diabetes_split <- initial_split(diabetes_clean, prop = 0.8, strata = readmitted)

train <- training(diabetes_split)

test <- testing(diabetes_split)


model <- logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

recipe <- recipe(readmitted ~., data = train) %>% 
  step_novel(all_factor_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())


folds <- vfold_cv(train, v = 5, strata = readmitted)

wf <- workflow() %>% 
  add_model(model) %>% 
  add_recipe(recipe)

grid <- grid_regular(
  penalty(range = c(-5, 0)),
  mixture(range = c(0,1)), levels = 8
)

print(grid, n = 64)

results <- tune_grid(wf, 
                     resamples = folds, 
                     grid = grid, 
                     metrics = metric_set(pr_auc, sensitivity, f_meas))

#autoplot(results)
#results

best_params <- select_by_one_std_err(results, metric = "pr_auc", penalty)

final_wf <- finalize_workflow(wf, best_params)

final_fit <- last_fit(final_wf, diabetes_split)

metrics <- collect_metrics(final_fit)
metrics

collect_predictions(final_fit) %>% 
  pr_auc(truth = readmitted,
         .pred_1,
         event_level = "second")

#PR_AUC is 0.556 
#a non linear model might perform better
#alternately adding the hospital codes might improve performance
#the result is not too surprising as most of the variables in the data set
#are drugs and most of the patients are not on them

