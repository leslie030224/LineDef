library(tidyverse)
library(gridExtra)
library(cowplot)  
library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = '../output/figure/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- lapply(split(x, x$variable), function(df) {
    df <- df[, grep("value", names(df))]
    names(df) <- gsub(".value", "", names(df))
    return(df)
  })

  df <- do.call(cbind, tmp)

  if(reverse == TRUE) { 
    ranking <- (max(sk_esd(df)$group) - sk_esd(df)$group) + 1 
  } else { 
    ranking <- sk_esd(df)$group 
  }

  x$rank <- paste("Rank", ranking[as.character(x$variable)])
  return(x)
}

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'
  
  return(top.k)
}


prediction_dir = '../output/prediction/LineDef/within-release/'
original_prediction_dir = "./line-level-baseline/output/prediction/DeepLineDP/within-release/"

all_files = list.files(prediction_dir)
original_all_files = list.files(original_prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

original_df_all <- NULL

for(f in original_all_files)
{
  original_df <- read.csv(paste0(original_prediction_dir, f))
  original_df_all <- rbind(original_df_all, original_df)
}


# ---------------- Code for RQ1 -----------------------#

get.file.level.metrics = function(df.file)
{
  all.gt = df.file$file.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label
  
  confusion.mat = caret::confusionMatrix(factor(all.pred), reference = factor(all.gt))
  
  bal.acc = confusion.mat$byClass["Balanced Accuracy"]
  AUC = pROC::auc(all.gt, all.prob)

  all.pred[all.pred=="False"] = 0
  all.pred[all.pred=="True"] = 1
  all.gt[all.gt=="False"] = 0
  all.gt[all.gt=="True"] = 1

  # all.gt = as.numeric_version(all.gt)
  all.gt = as.numeric(all.gt)
  
  # all.pred = as.numeric_version(all.pred)
  all.pred = as.numeric(all.pred)
  
  MCC = mcc(all.gt, all.pred, cutoff = 0.5)
  
  if(is.nan(MCC))
  {
    MCC = 0
  }
  
  eval.result = c(AUC, MCC, bal.acc)
  
  return(eval.result)
}

get.file.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)
  
  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.test.rels = c()
  
  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))

    if(method.name == "DP")
    {
      df = as_tibble(df)
      df = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
      
      df = distinct(df)
    }

    file.level.result = get.file.level.metrics(df)
    
    AUC = file.level.result[1]
    MCC = file.level.result[2]
    bal.acc = file.level.result[3]
    
    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.test.rels = append(all.test.rels,f)
    
  }
  
  result.df = data.frame(all.auc,all.mcc,all.bal.acc)
  
  
  all.test.rels = str_replace(all.test.rels, ".csv", "")
  
  result.df$release = all.test.rels
  result.df$technique = method.name
  
  return(result.df)
}

get.file.level.eval.result.GCN = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)
  
  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.test.rels = c()
  
  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))

    df = as_tibble(df)
    df = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
    df = distinct(df)

    file.level.result = get.file.level.metrics(df)
    
    AUC = file.level.result[1]
    MCC = file.level.result[2]
    bal.acc = file.level.result[3]
    
    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.test.rels = append(all.test.rels,f)
    
  }
  
  result.df = data.frame(all.auc,all.mcc,all.bal.acc)
  
  
  all.test.rels = str_replace(all.test.rels, ".csv", "")
  
  result.df$release = all.test.rels
  result.df$technique = method.name
  
  return(result.df)
}

bi.lstm.prediction.dir = "../output/prediction/Bi-LSTM/"
cnn.prediction.dir = "../output/prediction/CNN/"
dbn.prediction.dir = "../output/prediction/DBN/"
bow.prediction.dir = "../output/prediction/BoW/"

bi.lstm.result = get.file.level.eval.result(bi.lstm.prediction.dir, "Bi.LSTM")
cnn.result = get.file.level.eval.result(cnn.prediction.dir, "CNN")
dbn.result = get.file.level.eval.result(dbn.prediction.dir, "DBN")
bow.result = get.file.level.eval.result(bow.prediction.dir, "BoW")
gcn.result = get.file.level.eval.result.GCN(prediction_dir, "LineDef")
baseline.result = get.file.level.eval.result.GCN(original_prediction_dir, "DeepLineDP")

all.result = rbind(bi.lstm.result, cnn.result, dbn.result, bow.result, gcn.result, baseline.result)
names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")

write.csv(all.result, file = "output/RQ2.csv", row.names = FALSE)

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)
auc.result[auc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)
mcc.result[mcc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)
bal.acc.result[bal.acc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

p_auc <- ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("AUC") + 
  xlab("") + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(paste0(save.fig.dir,"file-AUC.png"),plot = p_auc, width=4,height=3)

p_bal_acc <- ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Balance Accuracy") + 
  xlab("") + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(paste0(save.fig.dir,"file-Balance_Accuracy.png"), plot = p_bal_acc, width=4,height=3)

p_mcc <- ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + 
  geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  ylab("MCC") + 
  xlab("") + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(paste0(save.fig.dir, "file-MCC.png"), plot = p_mcc, width=4,height=3)

combined_plot <- grid.arrange(p_auc, p_bal_acc, p_mcc, ncol = 3)

# 保存合并后的图形
ggsave(paste0(save.fig.dir, "file_results.png"), plot = combined_plot, width = 12, height = 3)


# ---------------- Code for RQ1 -----------------------#

## prepare data for baseline
line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
line.ground.truth = distinct(line.ground.truth)

get.line.metrics.result = function(baseline.df, cur.df.file)
{
  baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))
  
  sorted = baseline.df.with.ground.truth %>% group_by(filename) %>% arrange(-line.score, .by_group = TRUE) %>% mutate(order = row_number())
  
  #IFA
  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename)  %>% top_n(1, -order)
  
  ifa.list = IFA$order
  
  total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  
  #Recall20%LOC
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
  
  recall.list = recall20LOC$recall20LOC
  
  #Effort20%Recall
  effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  
  effort.list = effort20Recall$effort20Recall
  
  result.df = data.frame(ifa.list, recall.list, effort.list)
  
  return(result.df)
}

all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

error.prone.result.dir = '../output/ErrorProne_result/'
ngram.result.dir = '../output/n_gram_result/'
rf.result.dir = '../output/RF-line-level-result/'

n.gram.result.df = NULL
error.prone.result.df = NULL
rf.result.df = NULL 

## get result from baseline
for(rel in all_eval_releases)
{  
  error.prone.result = read.csv(paste0(error.prone.result.dir,rel,'-line-lvl-result.txt'),quote="")
  
  # levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="False"] = 0
  # levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="True"] = 1
  
  error.prone.result$EP_prediction_result[error.prone.result$EP_prediction_result=="False"] = 0
  error.prone.result$EP_prediction_result[error.prone.result$EP_prediction_result=="True"] = 1
  # error.prone.result$EP_prediction_result = as.numeric(as.numeric_version(error.prone.result$EP_prediction_result))
  error.prone.result$EP_prediction_result = as.numeric(error.prone.result$EP_prediction_result)
  
  names(error.prone.result) = c("filename","test","line.number","line.score")
  
  n.gram.result = read.csv(paste0(ngram.result.dir,rel,'-line-lvl-result.txt'), quote = "", sep='\t')
  n.gram.result = select(n.gram.result, "file.name", "line.number",  "line.score")
  n.gram.result = distinct(n.gram.result)
  names(n.gram.result) = c("filename", "line.number", "line.score")
  
  rf.result = read.csv(paste0(rf.result.dir,rel,'-line-lvl-result.csv'))
  rf.result = select(rf.result, "filename", "line_number","line.score.pred")
  names(rf.result) = c("filename", "line.number", "line.score")
  
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
  
  n.gram.eval.result = get.line.metrics.result(n.gram.result, cur.df.file)
  
  error.prone.eval.result = get.line.metrics.result(error.prone.result, cur.df.file)
  
  rf.eval.result = get.line.metrics.result(rf.result, cur.df.file)
  
  n.gram.result.df = rbind(n.gram.result.df, n.gram.eval.result)
  error.prone.result.df = rbind(error.prone.result.df, error.prone.eval.result)
  rf.result.df = rbind(rf.result.df, rf.eval.result)
  
  print(paste0('finished ', rel))
  
}

calculate_metrics <- function(df_all, top_k = 1500) {
  # Set token.attention.score to 0 for lines identified as comments
  df_all[df_all$is.comment.line == "True",]$token.attention.score = 0
  
  # Get top-k tokens
  tmp.top.k <- get.top.k.tokens(df_all, top_k)
  
  # Merge dataframes
  merged_df_all <- merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)
  merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0
  
  # Summarize attention scores for lines identified as true positives
  sum_line_attn <- merged_df_all %>%
    filter(file.level.ground.truth == "True" & prediction.label == "True") %>%
    group_by(test, filename, is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
    summarize(attention_score = sum(token.attention.score), num_tokens = n())
  
  # Sort the summarized data
  sorted <- sum_line_attn %>%
    group_by(test, filename) %>%
    arrange(-attention_score, .by_group=TRUE) %>%
    mutate(order = row_number())
  
  # Calculate IFA (Identified First Attacker)
  IFA <- sorted %>%
    filter(line.level.ground.truth == "True") %>%
    group_by(test, filename) %>%
    top_n(1, -order)
  
  # Calculate total true positives
  total_true <- sorted %>%
    group_by(test, filename) %>%
    summarize(total_true = sum(line.level.ground.truth == "True"))
  
  # Calculate Recall20%LOC (Recall at 20% Lines of Code)
  recall20LOC <- sorted %>%
    group_by(test, filename) %>%
    mutate(effort = round(order/n(), digits = 2 )) %>%
    filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>%
    mutate(recall20LOC = correct_pred/total_true)
  
  # Calculate Effort20%Recall (Effort to Achieve 20% Recall)
  effort20Recall <- sorted %>%
    merge(total_true) %>%
    group_by(test, filename) %>%
    mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"),
           recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  
  # Prepare data for plotting
  deeplinedp.ifa <- IFA$order
  deeplinedp.recall <- recall20LOC$recall20LOC
  deeplinedp.effort <- effort20Recall$effort20Recall
  
  deepline.dp.line.result <- data.frame(deeplinedp.ifa, deeplinedp.recall, deeplinedp.effort)
  
  return(deepline.dp.line.result)
}

# Example usage:

deepline.dp.line.result <- calculate_metrics(df_all)

baseline.line.result <- calculate_metrics(original_df_all)

names(rf.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(n.gram.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(error.prone.result.df)  = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(deepline.dp.line.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(baseline.line.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
write.csv(deepline.dp.line.result, file = "output/RQ3.csv", row.names = FALSE)

rf.result.df$technique = 'RF'
n.gram.result.df$technique = 'N.gram'
error.prone.result.df$technique = 'ErrorProne'
deepline.dp.line.result$technique = 'LineDef'
baseline.line.result$technique = 'DeepLineDP'

all.line.result = rbind(rf.result.df, n.gram.result.df, error.prone.result.df, deepline.dp.line.result, baseline.line.result)
recall.result.df = select(all.line.result, c('technique', 'Recall20%LOC'))
ifa.result.df = select(all.line.result, c('technique', 'IFA'))
effort.result.df = select(all.line.result, c('technique', 'Effort@20%Recall'))

recall.result.df = preprocess(recall.result.df, FALSE)
ifa.result.df = preprocess(ifa.result.df, TRUE)
effort.result.df = preprocess(effort.result.df, TRUE)

p_recall <- ggplot(recall.result.df, aes(x=reorder(variable, -value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Recall@Top20%LOC") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-Recall@Top20LOC.png"), plot = p_recall, width = 4, height = 3)

p_effort <- ggplot(effort.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Effort@Top20%Recall") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-Effort@Top20Recall.png"), plot = p_effort, width = 4, height = 3)

p_ifa <- ggplot(ifa.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  coord_cartesian(ylim=c(0, 175)) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("IFA") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-IFA.png"), plot = p_ifa, width = 4, height = 3)

combined_plot <- grid.arrange(p_recall, p_effort, p_ifa, ncol = 3)

ggsave(paste0(save.fig.dir, "line_results.png"), plot = combined_plot, width = 12, height = 3)


# ---------------- Code for RQ2 -----------------------#

## get within-project result
gcn.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")

file.level.by.project = gcn.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc))

names(file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy")

IFA$project = str_replace(IFA$test, '-.*','')
recall20LOC$project = str_replace(recall20LOC$test, '-.*','')
recall20LOC$project = as.factor(recall20LOC$project)
effort20Recall$project = str_replace(effort20Recall$test, '-.*','')

ifa.each.project = IFA %>% group_by(project) %>% summarise(mean.by.project = mean(order))
recall.each.project = recall20LOC %>% group_by(project) %>% summarise(mean.by.project = mean(recall20LOC))
effort.each.project = effort20Recall %>% group_by(project) %>% summarise(mean.by.project = mean(effort20Recall))

line.level.all.mean.by.project = data.frame(ifa.each.project$project, ifa.each.project$mean.by.project, recall.each.project$mean.by.project, effort.each.project$mean.by.project)

names(line.level.all.mean.by.project) = c("project", "IFA", "Recall20%LOC", "Effort@20%Recall")

write.csv(file.level.by.project, file = "output/within1.csv", row.names = FALSE)

write.csv(line.level.all.mean.by.project, file = "output/within2.csv", row.names = FALSE)

## get cross-project result

prediction.dir = '../output/prediction/DeepLineDP/cross-project/'

projs = c('activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket')

get.line.level.metrics = function(df_all)
{
  #Force attention score of comment line is 0
  df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

  sum_line_attn = df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
    summarize(attention_score = sum(token.attention.score), num_tokens = n())
  sorted = sum_line_attn %>% group_by(filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())

  # calculate IFA
  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename) %>% top_n(1, -order)
  total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))

  # calculate Recall20%LOC
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

  # calculate Effort20%Recall
  effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())

  all.ifa = IFA$order
  all.recall = recall20LOC$recall20LOC
  all.effort = effort20Recall$effort20Recall

  result.df = data.frame(all.ifa, all.recall, all.effort)

  return(result.df)
}


all.line.result = NULL
all.file.result = NULL


for(p in projs)
{
  actual.pred.dir = paste0(prediction.dir,p,'/')

  all.files = list.files(actual.pred.dir)

  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.src.projs = c()
  all.tar.projs = c()

  for(f in all.files)
  {
    df = read.csv(paste0(actual.pred.dir,f))

    f = str_replace(f,'.csv','')
    f.split = unlist(strsplit(f,'-'))
    target = tail(f.split,2)[1]

    df = as_tibble(df)

    df.file = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))

    df.file = distinct(df.file)
    if (nrow(df.file %>% filter(file.level.ground.truth == "True" & prediction.label == "True")) > 0) {
      # print(paste0('now is file level', f))
      file.level.result = get.file.level.metrics(df.file)

      AUC = file.level.result[1]
      MCC = file.level.result[2]
      bal.acc = file.level.result[3]

      all.auc = append(all.auc, AUC)
      all.mcc = append(all.mcc, MCC)
      all.bal.acc = append(all.bal.acc, bal.acc)

      all.src.projs = append(all.src.projs, p)
      all.tar.projs = append(all.tar.projs, target)

      tmp.top.k = get.top.k.tokens(df, 1500)

      merged_df_all = merge(df, tmp.top.k, by = c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

      merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0
      print(paste0('now is line level', f))
      line.level.result = get.line.level.metrics(merged_df_all)
      line.level.result$src = p
      line.level.result$target = target

      all.line.result = rbind(all.line.result, line.level.result)

      # print(paste0('finished ', f))
    } else {
      print(paste0('Skipping ', f, ' as no relevant data found.'))
    }
  }
  file.level.result = data.frame(all.auc,all.mcc,all.bal.acc)
  file.level.result$src = p
  file.level.result$target = all.tar.projs
  all.file.result = rbind(all.file.result, file.level.result)
  print(paste0('finished ',p))
}

final.file.level.result = all.file.result %>% group_by(target) %>% summarize(auc = mean(all.auc), balance_acc = mean(all.bal.acc), mcc = mean(all.mcc))

final.line.level.result = all.line.result %>% group_by(target) %>% summarize(recall = mean(all.recall), effort = mean(all.effort), ifa = mean(all.ifa))

write.csv(final.line.level.result, file = "output/cross1.csv", row.names = FALSE)

write.csv(final.file.level.result, file = "output/cross2.csv", row.names = FALSE)

