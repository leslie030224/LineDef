library(tidyverse)
library(gridExtra)
library(cowplot)  
library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = './output/figure/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- lapply(split(x, x$variable), function(df) {
    # Filter columns containing "value"
    df <- df[, grep("value", names(df))]
    # Rename columns by removing ".value"
    names(df) <- gsub(".value", "", names(df))
    return(df)
  })
  # Combine the list of data frames into a single data frame
  df <- do.call(cbind, tmp)
  
  # Compute ranking based on sk_esd function
  if(reverse == TRUE) { 
    ranking <- (max(sk_esd(df)$group) - sk_esd(df)$group) + 1 
  } else { 
    ranking <- sk_esd(df)$group 
  }
  
  # Add ranking as a new column to the original data frame
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


prediction_dir = '../../output/prediction/LineDef/within-release/'
gat.prediction.dir = "./GAT/output/prediction/LineDef/within-release/"
tagconv.prediction.dir = "./TAGConv/output/prediction/LineDef/within-release/"
sgconv.prediction.dir = "./SGConv/output/prediction/LineDef/within-release/"
ssgconv.prediction.dir = "./SSGConv/output/prediction/LineDef/within-release/"

all_files = list.files(prediction_dir)
gat_all_files = list.files(gat.prediction.dir)
tagconv_all_files = list.files(tagconv.prediction.dir)
sgconv_all_files = list.files(sgconv.prediction.dir)
ssgconv_all_files = list.files(ssgconv.prediction.dir)

df_all <- NULL
gat_all <- NULL
tagconv_all <- NULL
sgconv_all <- NULL
ssgconv_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

for(f in all_files)
{
  df1 <- read.csv(paste0(gat.prediction.dir, f))
  gat_all <- rbind(gat_all, df1)
}

for(f in all_files)
{
  df2 <- read.csv(paste0(tagconv.prediction.dir, f))
  tagconv_all <- rbind(tagconv_all, df2)
}

for(f in all_files)
{
  df3 <- read.csv(paste0(sgconv.prediction.dir, f))
  sgconv_all <- rbind(sgconv_all, df3) 
}

for(f in all_files)
{
  df4 <- read.csv(paste0(ssgconv.prediction.dir, f))
  ssgconv_all <- rbind(ssgconv_all, df4)
}

# ---------------- Code for RQ2 -----------------------#

get.file.level.metrics = function(df.file)
{
  all.gt = df.file$file.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label
  
  confusion.mat = caret::confusionMatrix(factor(all.pred), reference = factor(all.gt))
  # confusion.mat = confusionMatrix(all.pred, reference = all.gt)
  
  bal.acc = confusion.mat$byClass["Balanced Accuracy"]
  AUC = pROC::auc(all.gt, all.prob)
  
  # levels(all.pred)[levels(all.pred)=="False"] = 0
  # levels(all.pred)[levels(all.pred)=="True"] = 1
  # levels(all.gt)[levels(all.gt)=="False"] = 0
  # levels(all.gt)[levels(all.gt)=="True"] = 1
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

gat.result = get.file.level.eval.result(gat.prediction.dir, "GATConv")
tagconv.result = get.file.level.eval.result(tagconv.prediction.dir, "TAGConv")
sgconv.result = get.file.level.eval.result(sgconv.prediction.dir, "SGConv")
ssgconv.result = get.file.level.eval.result(ssgconv.prediction.dir, "SSGConv")
gcn.result = get.file.level.eval.result(prediction_dir, "GCNConv")

all.result = rbind(gat.result, tagconv.result, sgconv.result, ssgconv.result, gcn.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")

write.csv(all.result, file = "output/RQ2.csv", row.names = FALSE)

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)
# auc.result[auc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)
# mcc.result[mcc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)
# bal.acc.result[bal.acc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

p_auc <- ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("AUC") + 
  xlab("") + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir,"file-AUC.pdf"), plot=p_auc, width=4,height=3)

p_bal_acc <- ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() +
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Balance Accuracy") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir,"file-Balance_Accuracy.pdf"), plot=p_bal_acc, width=4,height=3)

p_mcc <- ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("MCC") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-MCC.pdf"), plot=p_mcc, width=4,height=3)

combined_plot <- grid.arrange(p_auc, p_bal_acc, p_mcc, ncol = 3)

# 保存合并后的图形
ggsave(paste0(save.fig.dir, "file_results.pdf"), plot = combined_plot, width = 12, height = 3)

# ---------------- Code for RQ3 -----------------------#

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

gcn.result <- calculate_metrics(df_all)
gat.result <- calculate_metrics(gat_all)
tagconv.result <- calculate_metrics(tagconv_all)
sgconv.result <- calculate_metrics(sgconv_all)
ssgconv.result <- calculate_metrics(ssgconv_all)


names(ssgconv.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(sgconv.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(tagconv.result)  = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(gat.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(gcn.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
write.csv(gcn.result, file = "output/RQ3.csv", row.names = FALSE)

tagconv.result$technique = 'TAGConv'
ssgconv.result$technique = 'SSGConv'
sgconv.result$technique = 'SGConv'
gat.result$technique = 'GATConv'
gcn.result$technique = 'LineDef'

all.line.result = rbind(tagconv.result, ssgconv.result, sgconv.result, gat.result, gcn.result)
recall.result.df = select(all.line.result, c('technique', 'Recall20%LOC'))
ifa.result.df = select(all.line.result, c('technique', 'IFA'))
effort.result.df = select(all.line.result, c('technique', 'Effort@20%Recall'))

recall.result.df = preprocess(recall.result.df, FALSE)
ifa.result.df = preprocess(ifa.result.df, TRUE)
effort.result.df = preprocess(effort.result.df, TRUE)

# 绘制 Recall@Top20%LOC 图
p_recall <- ggplot(recall.result.df, aes(x=reorder(variable, -value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Recall@Top20%LOC") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(paste0(save.fig.dir, "file-Recall@Top20LOC.pdf"), plot = p_recall, width = 4, height = 2.5)

# 绘制 Effort@Top20%Recall 图
p_effort <- ggplot(effort.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Effort@Top20%Recall") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-Effort@Top20Recall.pdf"), plot = p_effort, width = 4, height = 2.5)

# 绘制 IFA 图
p_ifa <- ggplot(ifa.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  coord_cartesian(ylim=c(0, 175)) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("IFA") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-IFA.png"), plot = p_ifa, width = 4, height = 2.5)

combined_plot <- grid.arrange(p_recall, p_effort, p_ifa, ncol = 3)

# 保存合并后的图形
ggsave(paste0(save.fig.dir, "line_results.pdf"), plot = combined_plot, width = 12, height = 3)