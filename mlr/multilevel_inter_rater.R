library(tidyverse)
library(lme4)
library(MuMIn)
library(lmerTest)


# functino to calculate icc
icc = function(model){
  #### compute ICC
  var.components = as.data.frame(VarCorr(model))$vcov
  ICC = var.components[1]/sum(var.components)

  #### find out average cluster size
  id.name = names(coef(model))
  clusters = nrow(matrix(unlist((coef(model)[id.name]))))
  n = length(residuals(model))
  average.cluster.size = n/clusters

  #### compute design effects
  design.effect = 1+ICC*(average.cluster.size-1)

  #### return stuff
  list(icc=ICC, design.effect=design.effect)

}


# data
data = read_csv('./complete_inter_metric.csv')

continous_vars = c('log_frequency',
                   'log_n_senses',
                   'nn_sim',
                   'l2_norm',
                   'dispersion_ave',
                   'es')

# mutate the data
data_centered = data %>%
  mutate(nn_sim = if_else(model == 'glove', glove_nn_sim, sgns_nn_sim),
         l2_norm = if_else(model == 'glove', glove_l2_norm, sgns_l2_norm),
         dispersion_ave = (sgns_dispersion+glove_dispersion)/2,
         es = if_else(model == 'glove', glove_es, sgns_es),
         log_frequency = log(frequency),
         log_n_senses = log(n_senses)) %>%
  mutate_at(continous_vars, scale, scale=FALSE) %>%
  mutate(log_frequency_squared = log_frequency^2)


# standardise the data
data_std = data %>%
  mutate(nn_sim = if_else(model == 'glove', glove_nn_sim, sgns_nn_sim),
         l2_norm = if_else(model == 'glove', glove_l2_norm, sgns_l2_norm),
         dispersion_ave = (sgns_dispersion+glove_dispersion)/2,
         es = if_else(model == 'glove', glove_es, sgns_es),
         log_frequency = log(frequency),
         log_n_senses = log(n_senses)) %>%
  mutate_at(continous_vars, scale, scale=TRUE) %>%
  mutate(log_frequency_squared = log_frequency^2)


#ICC intercept only model
model_intercept = lmer('score ~ 1 + (1 | model) + (1 | corpus)',
                       control = lmerControl(optimizer ="Nelder_Mead"),
                       data = data)

summary(model_intercept)

icc(model_intercept) #icc: 0.07211365 (i think only for one random effect here)
r.squaredGLMM(model_intercept) #conditional R2: 0.4535211
var.components = as.data.frame(VarCorr(model_intercept))$vcov
ICC = var.components[2]/sum(var.components)
ICC

# explained variance by corpus: 0.07211365
# explained variance by model: 0.3814074


# check multicollinearity (pearson's R) (result similar to spearman)
cor(select(data_std, log_frequency, log_frequency_squared,
           log_n_senses, nn_sim, l2_norm, dispersion_ave, es, score), method = "pearson", use = "complete.obs")
# log_frequency with l2_norm, ~.75
# log_frequency with dispersion_ave, ~.90
# we should probably just leave out dispersion_ave


# scatter plot: log_frequency vs. score - relationship seems quite mixed across groups
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = log_frequency, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')

# scatter plot: log_frequency_squared vs. score - relationship seems quite mixed across groups
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = log_frequency_squared, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')

# scatter plot: log_n_senses vs. score - relationship seems quite mixed across groups
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = log_n_senses, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')

# scatter plot: nn_sim vs. score - relationship seems overall positive across groups
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = nn_sim, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')

# scatter plot: l2_norm vs. score - relationship seems very mixed across groups
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = l2_norm, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')

# scatter plot: dispersion_ave vs. score - relationship seems positive except for one group
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = dispersion_ave, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')

# scatter plot: es vs. score - relationship seems very mixed across groups
set.seed(123)
data_std %>%
  sample_frac(0.2) %>%
  ggplot(aes(x = es, y = score)) +
  geom_point() +
  facet_wrap(~model*corpus) +
  geom_smooth(method='lm')


#full model (only centered by not standardized)
model_full_raw = lmer('score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)',
                      control = lmerControl(optimizer ="Nelder_Mead"),
                      data = data_centered)

summary(model_full_raw)

#full model (centered and standardized)
model_full_std = lmer('score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)',
                      control = lmerControl(optimizer ="Nelder_Mead"),
                      data = data_std)

summary(model_full_std)
r.squaredGLMM(model_full_std) #R2M: 0.05813408 R2C: 0.5159669
# the residual of the intercept-only model is: 0.0075220
var.components = as.data.frame(VarCorr(model_full_std))$vcov
ICC = var.components[2]/(sum(var.components)+0.0075220-var.components[3])
ICC

(r.squaredGLMM(model_full_std)[2]-r.squaredGLMM(model_full_std)[1])*var.components[1]/(sum(var.components[c(1,2)]))
(r.squaredGLMM(model_full_std)[2]-r.squaredGLMM(model_full_std)[1])*var.components[2]/(sum(var.components[c(1,2)]))
# 0.027122(corpus) and 0.4307108 (model)


# ablation study
formula_full = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)'

formula_minus_log_frequency = 'score ~ 1 + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)'

formula_minus_log_frequency_squared = 'score ~ 1 + log_frequency + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)'

formula_minus_log_n_senses = 'score ~ 1 + log_frequency + log_frequency_squared + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)'

formula_minus_pos = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses +
                      nn_sim + l2_norm + es + (1 | model) + (1 | corpus)'

formula_minus_nn_sim = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos + l2_norm + es + (1 | model) + (1 | corpus)'

formula_minus_l2_norm = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + es + (1 | model) + (1 | corpus)'

formula_minus_es = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + (1 | model) + (1 | corpus)'

formula_minus_model = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | corpus)'

formula_minus_corpus = 'score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                      nn_sim + l2_norm + es + (1 | model)'

all_formulas = list("log_frequency" = formula_minus_log_frequency,
                    "log_frequency_squared" = formula_minus_log_frequency_squared,
                    "log_n_senses" = formula_minus_log_n_senses,
                    "pos" = formula_minus_pos,
                    "nn_sim" = formula_minus_nn_sim,
                    "l2_norm" = formula_minus_l2_norm,
                    "es" = formula_minus_es,
                    "model" = formula_minus_model,
                    "corpus" = formula_minus_corpus)

ablation_r2_ls = list()
for (i in 1:length(all_formulas)) {
  predictor = names(all_formulas[i])
  formula_foo = all_formulas[[i]]

  model_ablation = lmer(formula = formula_foo,
                        control = lmerControl(optimizer ="Nelder_Mead"),
                        data = data_std)

  model_r2 = r.squaredGLMM(model_ablation)

  ablation_r2_ls[[predictor]] = model_r2
}

#R2M: 0.05813408 R2C: 0.5159669
do.call(rbind.data.frame, ablation_r2_ls) %>%
  mutate(var = row.names(.)) %>%
  as_tibble() %>%
  mutate(R2c_change = 0.5159669 - R2c,
         R2m_change = 0.05813408 - R2m) %>%
  mutate(R2c_change = round(R2c_change, 4),
         R2m_change = round(R2m_change, 4))


#interaction analysis: between log_frequency and other properties
model_interact_std = lmer('score ~ 1 + log_frequency + log_frequency_squared + log_n_senses + most_common_pos +
                          nn_sim + l2_norm + es +
                          log_frequency*log_n_senses + log_frequency*nn_sim + log_frequency*l2_norm + log_frequency*es + log_frequency*most_common_pos +
                          (1 | model) + (1 | corpus)',
                          control = lmerControl(optimizer ="Nelder_Mead"),
                          data = data_std)

summary(model_interact_std)
r.squaredGLMM(model_interact_std)
